"""
Crowd Prediction Training Module

Train models (Random Forest, XGBoost, Ridge, etc.) using weather and temporal features
to predict beach crowd counts.

Usage:
    from training_module import train_pipeline, compare_models, load_predictions_from_cache

    # From cache directory (your structure)
    results = train_pipeline(
        "cache/predictions",
        counting_model="bayesian_vgg19",  # filter by counting model
        model_type="random_forest",       # or xgboost if installed
        save_name="crowd_model"
    )

    # From JSON file
    results = train_pipeline("predictions.json", model_type="random_forest")

    # Explore cache
    from training_module import summarize_cache
    summary = summarize_cache("cache/predictions", model="bayesian_vgg19")
    print(summary)

    # Step by step
    df = load_predictions_from_cache("cache/predictions", model="bayesian_vgg19")
    X, y, features = prepare_data(df)
    comparison = compare_models(X, y)

CLI:
    python training_module.py cache/predictions -c bayesian_vgg19 --save my_model
    python training_module.py cache/predictions -c bayesian_vgg19 --compare
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

PROJECT_ROOT = Path.cwd()
MODELS_DIR = PROJECT_ROOT / 'models/prediction'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_predictions_from_json(filepath: str) -> pd.DataFrame:
    """Load predictions from JSON file (task results or API response)."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    if 'results' in data:
        records = data['results']
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unknown JSON format")

    return _records_to_dataframe(records)


def load_predictions_from_cache(cache_dir: str, model: str = None, beach: str = None) -> pd.DataFrame:
    """
    Load predictions from cache directory.

    Supports structures:
    - cache/predictions/{model}/{beach}/{hash}.json
    - cache/predictions/{beach}/{hash}.json
    - any folder with .json files

    Args:
        cache_dir: Path to cache directory (e.g., "cache/predictions")
        model: Filter by model name (e.g., "bayesian_vgg19")
        beach: Filter by beach folder (e.g., "HeliosHotel_frontline")

    Returns:
        DataFrame with all predictions
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        raise ValueError(f"Cache directory not found: {cache_dir}")

    # Build search path
    if model:
        model_path = cache_path / model
        if model_path.exists():
            cache_path = model_path

    if beach:
        beach_path = cache_path / beach
        if beach_path.exists():
            cache_path = beach_path

    records = []
    json_files = list(cache_path.rglob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {cache_path}")

    print(f"Found {len(json_files)} JSON files in {cache_path}")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                record = json.load(f)

            # Extract beach info from path if not in record
            if 'beach_folder' not in record or not record['beach_folder']:
                try:
                    rel_path = json_file.parent.relative_to(cache_path)
                    record['beach_folder'] = str(rel_path) if str(rel_path) != '.' else json_file.parent.name
                except:
                    record['beach_folder'] = json_file.parent.name

            records.append(record)
        except Exception as e:
            pass  # Skip invalid files

    if not records:
        raise ValueError(f"No valid records found in {cache_path}")

    print(f"Loaded {len(records)} valid records")

    return _records_to_dataframe(records)


def load_from_path(path: str, model: str = None, beach: str = None) -> pd.DataFrame:
    """
    Smart loader - detects if path is a file or directory and loads accordingly.

    Args:
        path: Path to JSON file OR cache directory
        model: Filter by model (for cache dirs)
        beach: Filter by beach (for cache dirs)

    Returns:
        DataFrame with predictions
    """
    p = Path(path)

    if p.is_file():
        return load_predictions_from_json(str(p))
    elif p.is_dir():
        return load_predictions_from_cache(str(p), model=model, beach=beach)
    else:
        raise ValueError(f"Path not found: {path}")


def summarize_cache(cache_dir: str) -> pd.DataFrame:
    """
    Get summary of what's in a cache directory.

    Returns DataFrame with columns: model, beach, count, date_range
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        raise ValueError(f"Cache directory not found: {cache_dir}")

    # Find all JSON files
    json_files = list(cache_path.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found in {cache_dir}")
        return pd.DataFrame()

    # Group by parent directories
    summary = {}

    for json_file in json_files:
        try:
            # Get relative path parts
            rel_path = json_file.relative_to(cache_path)
            parts = rel_path.parts

            if len(parts) >= 2:
                model = parts[0]
                beach = '/'.join(parts[1:-1]) if len(parts) > 2 else parts[1] if len(parts) > 1 else 'root'
            else:
                model = 'unknown'
                beach = json_file.parent.name

            key = (model, beach)
            if key not in summary:
                summary[key] = {'files': 0, 'dates': []}

            summary[key]['files'] += 1

            # Try to get datetime from file
            with open(json_file, 'r') as f:
                record = json.load(f)
                if 'datetime' in record:
                    summary[key]['dates'].append(record['datetime'])
        except:
            pass

    # Convert to DataFrame
    rows = []
    for (model, beach), info in summary.items():
        row = {
            'model': model,
            'beach': beach,
            'count': info['files'],
        }
        if info['dates']:
            dates = pd.to_datetime(info['dates'])
            row['date_min'] = dates.min()
            row['date_max'] = dates.max()
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(['model', 'beach']).reset_index(drop=True)

    return df


def _records_to_dataframe(records: List[Dict]) -> pd.DataFrame:
    """Convert prediction records to DataFrame with flattened weather."""
    rows = []
    for r in records:
        if 'error' in r:
            continue

        row = {
            'filename': r.get('filename'),
            'beach': r.get('beach') or r.get('name'),
            'beach_folder': r.get('beach_folder') or r.get('cache_name'),
            'datetime': r.get('datetime'),
            'lat': r.get('lat'),
            'lon': r.get('lon'),
            'count': r.get('count'),
            'model': r.get('model')
        }

        weather = r.get('weather', {})
        if weather:
            for k, v in weather.items():
                row[k] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from datetime."""
    df = df.copy()

    if 'datetime' not in df.columns:
        return df

    dt = df['datetime']
    df['hour'] = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek
    df['day_of_year'] = dt.dt.dayofyear
    df['month'] = dt.dt.month
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def get_feature_columns(df: pd.DataFrame,
                        include_weather: bool = True,
                        include_temporal: bool = True,
                        include_location: bool = False,
                        custom_features: List[str] = None) -> List[str]:
    """Get feature column names based on configuration."""
    features = []

    if include_weather:
        weather_cols = [c for c in df.columns if c.startswith('ae_') or c.startswith('om_')]
        features.extend(weather_cols)

    if include_temporal:
        temporal_cols = ['hour', 'day_of_week', 'month', 'is_weekend',
                         'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                         'dow_sin', 'dow_cos']
        features.extend([c for c in temporal_cols if c in df.columns])

    if include_location:
        if 'lat' in df.columns:
            features.append('lat')
        if 'lon' in df.columns:
            features.append('lon')

    if custom_features:
        features.extend([f for f in custom_features if f in df.columns])

    return list(dict.fromkeys(features))  # Remove duplicates


def prepare_data(df: pd.DataFrame,
                 target: str = 'count',
                 features: List[str] = None,
                 include_weather: bool = True,
                 include_temporal: bool = True,
                 include_location: bool = False,
                 custom_features: List[str] = None) -> tuple:
    """Prepare X and y for training."""
    df = add_temporal_features(df)

    if features is None:
        features = get_feature_columns(df, include_weather, include_temporal,
                                       include_location, custom_features)

    features = [f for f in features if f in df.columns]

    if not features:
        raise ValueError("No valid features found")

    df_clean = df.dropna(subset=features + [target])

    X = df_clean[features].copy()
    y = df_clean[target].copy()

    return X, y, features


# ============================================================
# MODEL REGISTRY
# ============================================================

def get_models() -> Dict[str, Any]:
    """Get available model constructors."""
    models = {
        'random_forest': lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        'gradient_boosting': lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
        'ridge': lambda: Ridge(alpha=1.0),
        'lasso': lambda: Lasso(alpha=1.0),
        'elastic_net': lambda: ElasticNet(alpha=1.0, l1_ratio=0.5),
        'svr': lambda: SVR(kernel='rbf', C=1.0),
    }

    if HAS_XGB:
        models['xgboost'] = lambda: xgb.XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42, verbosity=0)

    if HAS_LGB:
        models['lightgbm'] = lambda: lgb.LGBMRegressor(n_estimators=100, n_jobs=-1, random_state=42, verbose=-1)

    return models


def list_available_models() -> List[str]:
    """List available model types."""
    return list(get_models().keys())


# ============================================================
# TRAINING
# ============================================================

def train_model(X: pd.DataFrame, y: pd.Series,
                model_type: str = 'random_forest',
                test_size: float = 0.2,
                scale: bool = False,
                time_series_split: bool = False) -> Dict[str, Any]:
    """Train a single model and return results."""
    models = get_models()
    if model_type not in models:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(models.keys())}")

    # Split
    if time_series_split:
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scale
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    # Train
    model = models[model_type]()
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = {
        'model_type': model_type,
        'model': model,
        'scaler': scaler,
        'features': list(X.columns),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'metrics': {
            'train': {
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test': {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            }
        }
    }

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=X.columns)
        results['feature_importance'] = importance.sort_values(ascending=False).to_dict()
    elif hasattr(model, 'coef_'):
        importance = pd.Series(np.abs(model.coef_), index=X.columns)
        results['feature_importance'] = importance.sort_values(ascending=False).to_dict()

    return results


def compare_models(X: pd.DataFrame, y: pd.Series,
                   models: List[str] = None,
                   cv_folds: int = 5,
                   time_series_cv: bool = False) -> pd.DataFrame:
    """Compare multiple models using cross-validation."""
    available = get_models()
    if models is None:
        models = list(available.keys())

    if time_series_cv:
        cv = TimeSeriesSplit(n_splits=cv_folds)
    else:
        cv = cv_folds

    results = []
    for model_type in models:
        if model_type not in available:
            continue

        model = available[model_type]()

        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
            rmse_scores = -scores

            results.append({
                'model': model_type,
                'rmse_mean': rmse_scores.mean(),
                'rmse_std': rmse_scores.std(),
                'rmse_min': rmse_scores.min(),
                'rmse_max': rmse_scores.max()
            })
            print(f"  {model_type}: RMSE = {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
        except Exception as e:
            results.append({'model': model_type, 'error': str(e)})
            print(f"  {model_type}: ERROR - {e}")

    return pd.DataFrame(results).sort_values('rmse_mean', na_position='last')


# ============================================================
# MODEL PERSISTENCE
# ============================================================

def save_model(results: Dict[str, Any], name: str, save_dir: str = None) -> str:
    """Save trained model to disk."""
    if save_dir:
        save_path = Path(save_dir)
    else:
        save_path = MODELS_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / f"{name}.pkl"

    save_data = {
        'model': results['model'],
        'scaler': results.get('scaler'),
        'features': results['features'],
        'model_type': results['model_type'],
        'metrics': results['metrics'],
        'feature_importance': results.get('feature_importance'),
        'saved_at': datetime.now().isoformat()
    }

    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)

    return str(filepath)


def load_model(filepath: str) -> Dict[str, Any]:
    """Load trained model from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def predict_with_model(model_data: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Make predictions using loaded model."""
    model = model_data['model']
    scaler = model_data.get('scaler')
    features = model_data['features']

    X_pred = X[features].copy()

    if scaler:
        X_pred = pd.DataFrame(scaler.transform(X_pred), columns=features, index=X_pred.index)

    return model.predict(X_pred)


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_pipeline(data_source: str,
                   model_type: str = 'random_forest',
                   counting_model: str = None,
                   beach_filter: str = None,
                   include_weather: bool = True,
                   include_temporal: bool = True,
                   include_location: bool = False,
                   custom_features: List[str] = None,
                   test_size: float = 0.2,
                   time_series_split: bool = True,
                   save_name: str = None,
                   save_dir: str = None) -> Dict[str, Any]:
    """
    Full training pipeline from data source to saved model.

    Args:
        data_source: Path to JSON file or cache directory
        model_type: ML model to train (xgboost, random_forest, etc.)
        counting_model: Filter by counting model (e.g., "bayesian_vgg19")
        beach_filter: Filter to specific beach (optional)
        include_weather: Use weather features
        include_temporal: Use temporal features (hour, day, month...)
        include_location: Use lat/lon features
        custom_features: Additional feature column names
        test_size: Test set proportion
        time_series_split: Use chronological split (recommended)
        save_name: Name to save model (optional)
        save_dir: Directory to save model (optional)

    Returns:
        Training results dict with model, metrics, features

    Examples:
        # From JSON file
        train_pipeline("predictions.json", model_type="random_forest")

        # From cache directory
        train_pipeline("cache/predictions", counting_model="bayesian_vgg19")

        # Filter by beach
        train_pipeline("cache/predictions", beach_filter="HeliosHotel_frontline")
    """
    source = Path(data_source)

    # Load data
    if source.is_file() and source.suffix == '.json':
        print(f"Loading from JSON file: {source}")
        df = load_predictions_from_json(str(source))
    else:
        print(f"Loading from cache directory: {source}")
        df = load_predictions_from_cache(str(source), model=counting_model, beach=beach_filter)
        beach_filter = None  # Already filtered during load

    print(f"Total records: {len(df)}")

    # Show beaches
    if 'beach' in df.columns:
        beaches = df['beach'].value_counts()
        print(f"\nBeaches ({len(beaches)}):")
        for beach, count in beaches.head(10).items():
            print(f"  {beach}: {count}")
        if len(beaches) > 10:
            print(f"  ... and {len(beaches) - 10} more")

    # Filter by beach
    if beach_filter:
        col = 'beach' if 'beach' in df.columns else 'beach_folder'
        df = df[df[col].str.contains(beach_filter, case=False, na=False)]
        print(f"\nFiltered to {len(df)} records matching: {beach_filter}")

    if len(df) < 10:
        raise ValueError(f"Not enough data: {len(df)} records (need at least 10)")

    # Prepare features
    X, y, features = prepare_data(
        df,
        include_weather=include_weather,
        include_temporal=include_temporal,
        include_location=include_location,
        custom_features=custom_features
    )

    print(f"\nFeatures ({len(features)}):")
    for i, f in enumerate(features[:15]):
        print(f"  {f}")
    if len(features) > 15:
        print(f"  ... and {len(features) - 15} more")

    print(f"\nSamples: {len(X)}")
    print(f"Target range: {y.min():.1f} - {y.max():.1f} (mean: {y.mean():.1f})")

    # Train
    results = train_model(X, y, model_type=model_type, test_size=test_size, time_series_split=time_series_split)

    # Print results
    print(f"\n{'='*50}")
    print(f"{model_type.upper()} RESULTS")
    print(f"{'='*50}")
    print(f"Train: RMSE={results['metrics']['train']['rmse']:.2f}, MAE={results['metrics']['train']['mae']:.2f}, R²={results['metrics']['train']['r2']:.3f}")
    print(f"Test:  RMSE={results['metrics']['test']['rmse']:.2f}, MAE={results['metrics']['test']['mae']:.2f}, R²={results['metrics']['test']['r2']:.3f}")

    if 'feature_importance' in results:
        print(f"\nTop 10 Features:")
        for i, (feat, imp) in enumerate(list(results['feature_importance'].items())[:10]):
            print(f"  {i+1}. {feat}: {imp:.4f}")

    # Save
    if save_name:
        filepath = save_model(results, save_name, save_dir)
        print(f"\nModel saved: {filepath}")
        results['saved_path'] = filepath

    return results


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Train crowd prediction models')
    parser.add_argument('data_source', help='Path to JSON file or cache directory')
    parser.add_argument('--model', '-m', default='random_forest',
                        choices=list(get_models().keys()),
                        help='ML model type')
    parser.add_argument('--counting-model', '-c', default=None,
                        help='Filter by counting model (e.g., bayesian_vgg19)')
    parser.add_argument('--beach', '-b', default=None,
                        help='Filter by beach name')
    parser.add_argument('--save', '-s', default=None,
                        help='Save model with this name')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all models')
    parser.add_argument('--no-weather', action='store_true',
                        help='Exclude weather features')
    parser.add_argument('--no-temporal', action='store_true',
                        help='Exclude temporal features')

    args = parser.parse_args()

    print(f"Available models: {list(get_models().keys())}")
    print()

    # Load data first if comparing
    if args.compare:
        source = Path(args.data_source)
        if source.is_file():
            df = load_predictions_from_json(str(source))
        else:
            df = load_predictions_from_cache(str(source), model=args.counting_model)

        if args.beach:
            col = 'beach' if 'beach' in df.columns else 'beach_folder'
            df = df[df[col].str.contains(args.beach, case=False, na=False)]

        X, y, features = prepare_data(
            df,
            include_weather=not args.no_weather,
            include_temporal=not args.no_temporal
        )

        print(f"Comparing models on {len(X)} samples with {len(features)} features...")
        print()
        comparison = compare_models(X, y, time_series_cv=True)
        print()
        print(comparison.to_string(index=False))
    else:
        results = train_pipeline(
            args.data_source,
            model_type=args.model,
            counting_model=args.counting_model,
            beach_filter=args.beach,
            include_weather=not args.no_weather,
            include_temporal=not args.no_temporal,
            save_name=args.save
        )