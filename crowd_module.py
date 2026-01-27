import sys
import io
import json
import hashlib
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod

PROJECT_ROOT = Path.cwd()
PREDICTION_CACHE_DIR = PROJECT_ROOT / 'cache/predictions'
PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ABSTRACT BASE CLASS
# ============================================================

class BasePredictor(ABC):
    """Base class for crowd counting models."""

    name: str = "base"

    @abstractmethod
    def load(self):
        """Load model weights."""
        pass

    @abstractmethod
    def predict(self, image_data) -> float:
        """Predict crowd count from image."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """Get model info."""
        pass

    def predict_with_density(self, image_data):
        """Override in subclass if density map is supported."""
        raise NotImplementedError("Density map not supported for this model")


# ============================================================
# BAYESIAN VGG19 IMPLEMENTATION
# ============================================================

class BayesianVGG19Predictor(BasePredictor):
    """Bayesian Crowd Counting with VGG19 backbone."""

    name = "bayesian_vgg19"

    def __init__(self, weights_path=None, repo_path=None):
        self.weights_path = Path(weights_path) if weights_path else PROJECT_ROOT / 'resources/best_model.pth'
        self.repo_path = Path(repo_path) if repo_path else PROJECT_ROOT / 'Bayesian-Crowd-Counting'
        self.model = None
        self.device = None
        self.transform = None

    def load(self):
        if self.model is not None:
            return

        import torch
        from torchvision import transforms

        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        from models.vgg import vgg19

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = vgg19()
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"[{self.name}] Model loaded on {self.device}")

    def predict(self, image_data) -> float:
        import torch
        from PIL import Image

        self.load()

        if isinstance(image_data, (str, Path)):
            img = Image.open(image_data).convert('RGB')
        else:
            img = Image.open(io.BytesIO(image_data)).convert('RGB')

        inputs = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        return torch.sum(outputs).item()

    def predict_with_density(self, image_data):
        import torch
        from PIL import Image

        self.load()

        if isinstance(image_data, (str, Path)):
            img = Image.open(image_data).convert('RGB')
        else:
            img = Image.open(io.BytesIO(image_data)).convert('RGB')

        inputs = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)

        count = torch.sum(outputs).item()
        density = outputs.squeeze().cpu().numpy()
        return count, density

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_info(self) -> dict:
        return {
            "name": self.name,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
            "weights_path": str(self.weights_path),
            "repo_path": str(self.repo_path)
        }


# ============================================================
# MODEL REGISTRY
# ============================================================

class PredictorRegistry:
    """Registry for managing multiple predictor models."""

    def __init__(self):
        self._predictors: dict[str, BasePredictor] = {}
        self._active: str = None

    def register(self, predictor: BasePredictor):
        """Register a predictor."""
        self._predictors[predictor.name] = predictor
        if self._active is None:
            self._active = predictor.name

    def set_active(self, name: str):
        """Set active predictor by name."""
        if name not in self._predictors:
            raise ValueError(f"Predictor '{name}' not registered. Available: {list(self._predictors.keys())}")
        self._active = name

    def get_active(self) -> BasePredictor:
        """Get active predictor."""
        if self._active is None:
            raise ValueError("No predictor registered")
        return self._predictors[self._active]

    def get(self, name: str) -> BasePredictor:
        """Get predictor by name."""
        if name not in self._predictors:
            raise ValueError(f"Predictor '{name}' not registered")
        return self._predictors[name]

    def list_predictors(self) -> dict:
        """List all registered predictors."""
        return {
            "active": self._active,
            "available": list(self._predictors.keys()),
            "predictors": {name: p.get_info() for name, p in self._predictors.items()}
        }

    def load_active(self):
        """Load the active predictor."""
        self.get_active().load()


# Global registry
_registry = PredictorRegistry()

# Register default predictor
_registry.register(BayesianVGG19Predictor())


# ============================================================
# PUBLIC API (backwards compatible)
# ============================================================

def predict_count(image_data) -> float:
    """Predict crowd count using active model."""
    return round(_registry.get_active().predict(image_data), 4)


def predict_count_with_density(image_data):
    """Predict crowd count and density map using active model."""
    return _registry.get_active().predict_with_density(image_data)


def load_model():
    """Load the active model."""
    _registry.load_active()


def get_model_info() -> dict:
    """Get info about all registered models."""
    return _registry.list_predictors()


def register_predictor(predictor: BasePredictor):
    """Register a new predictor."""
    _registry.register(predictor)


def set_active_predictor(name: str):
    """Set active predictor by name."""
    _registry.set_active(name)


def get_predictor(name: str) -> BasePredictor:
    """Get predictor by name."""
    return _registry.get(name)


# ============================================================
# CACHE FUNCTIONS
# ============================================================

def _get_prediction_cache_key(filename, lat, lon):
    key = f"{filename}_{lat:.4f}_{lon:.4f}"
    return hashlib.md5(key.encode()).hexdigest()


def _sanitize_name(name):
    return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)


def _get_cache_dir(model=None, name=None, subpath=None):
    """Get cache directory: cache/predictions/{model}/{name}/{subpath}/"""
    cache_dir = PREDICTION_CACHE_DIR
    if model:
        cache_dir = cache_dir / _sanitize_name(model)
    if name:
        cache_dir = cache_dir / _sanitize_name(name)
    if subpath:
        cache_dir = cache_dir / subpath
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_prediction_from_cache(filename, lat, lon, model=None, name=None, subpath=None):
    cache_key = _get_prediction_cache_key(filename, lat, lon)
    cache_dir = _get_cache_dir(model, name, subpath)
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None


def save_prediction_to_cache(filename, lat, lon, result, model=None, name=None, subpath=None):
    cache_key = _get_prediction_cache_key(filename, lat, lon)
    cache_dir = _get_cache_dir(model, name, subpath)
    cache_file = cache_dir / f"{cache_key}.json"
    with open(cache_file, 'w') as f:
        json.dump(result, f)


def clear_prediction_cache(model=None, name=None):
    """Clear cache. Can filter by model and/or name."""
    count = 0
    if model or name:
        cache_dir = _get_cache_dir(model, name)
        for f in cache_dir.rglob("*.json"):
            f.unlink()
            count += 1
    else:
        for f in PREDICTION_CACHE_DIR.rglob("*.json"):
            f.unlink()
            count += 1
    return count


def list_cache_names():
    """List cache structure: {model: {name: count}}"""
    result = {}

    for model_dir in PREDICTION_CACHE_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        model_data = {}

        direct_count = len(list(model_dir.glob("*.json")))
        if direct_count > 0:
            model_data["_root"] = direct_count

        for name_dir in model_dir.iterdir():
            if name_dir.is_dir():
                total = len(list(name_dir.rglob("*.json")))
                if total > 0:
                    subdirs = {}
                    for sub in name_dir.rglob("*"):
                        if sub.is_dir():
                            sub_count = len(list(sub.glob("*.json")))
                            if sub_count > 0:
                                rel = str(sub.relative_to(name_dir))
                                subdirs[rel] = sub_count

                    if subdirs:
                        model_data[name_dir.name] = {"_total": total, "subdirs": subdirs}
                    else:
                        model_data[name_dir.name] = total

        if model_data:
            result[model_name] = model_data

    return result


def parse_datetime_from_filename(filename):
    try:
        stem = Path(filename).stem
        ts = int(stem)
        return datetime.fromtimestamp(ts)
    except:
        return None


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    load_model()
    print(json.dumps(get_model_info(), indent=2))

    test_image = PROJECT_ROOT / "sample_images/1657879200.jpg"
    if test_image.exists():
        count = predict_count(test_image)
        print(f"Count: {count}")