# =============================================================================
# THREADING & MEMORY FIXES FOR PYTORCH
# Must be at the very top before other imports
# =============================================================================
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from time import sleep
import threading
import uuid
import json
import gc
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Query, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel
import uvicorn
import io
import csv

# PyTorch memory management
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def cleanup_memory():
    """Force cleanup of GPU/CPU memory after predictions."""
    gc.collect()
    if HAS_TORCH:
        # CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # MPS (Mac Metal) cleanup - more aggressive
        if hasattr(torch, 'mps') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Synchronize to ensure all operations complete
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            torch.mps.empty_cache()
    # Force Python gc multiple times (helps with circular refs)
    gc.collect()
    gc.collect()


from weather_module import get_weather_at_point, get_station_info, reload_aemet_data, check_openmeteo_connectivity
from crowd_module import (
    predict_count, load_model, get_model_info, parse_datetime_from_filename,
    load_prediction_from_cache, save_prediction_to_cache, clear_prediction_cache, list_cache_names,
    register_predictor, set_active_predictor, get_predictor, BasePredictor
)

# Training module imports
try:
    from training_module import (
        load_predictions_from_cache as load_training_data,
        load_from_path,
        summarize_cache,
        prepare_data,
        train_model,
        compare_models,
        save_model as save_trained_model,
        load_model as load_trained_model,
        predict_with_model,
        list_available_models,
        add_temporal_features
    )

    HAS_TRAINING = True
except ImportError:
    HAS_TRAINING = False


# ============================================================
# TASK MANAGER - Memory Efficient (no result storage)
# ============================================================

class TaskManager:
    """
    Memory-efficient task manager that does NOT store results in memory.
    Results are saved to cache and can be retrieved later.
    Only counters and progress are tracked.
    """

    def __init__(self):
        self._tasks = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)  # Reduced workers

    def create_task(self, task_type: str, params: dict) -> str:
        task_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._tasks[task_id] = {
                "id": task_id,
                "type": task_type,
                "status": "pending",
                "progress": 0,
                "total": 0,
                "current": 0,
                "params": params,
                "errors": 0,
                "success": 0,
                "from_cache": 0,
                "last_processed": None,
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "error": None
            }
        return task_id

    def get_task(self, task_id: str) -> dict:
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs):
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(kwargs)

    def increment_progress(self, task_id: str, success: bool = True, from_cache: bool = False, last_item: str = None):
        """Lightweight progress update - no result storage"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["current"] += 1
                total = self._tasks[task_id]["total"]
                if total > 0:
                    self._tasks[task_id]["progress"] = round(self._tasks[task_id]["current"] / total * 100, 1)
                if success:
                    self._tasks[task_id]["success"] += 1
                else:
                    self._tasks[task_id]["errors"] += 1
                if from_cache:
                    self._tasks[task_id]["from_cache"] += 1
                if last_item:
                    self._tasks[task_id]["last_processed"] = last_item

    def add_result(self, task_id: str, result: dict):
        """Legacy method - now just updates counters, doesn't store result"""
        is_error = "error" in result
        from_cache = result.get("from_cache", False)
        filename = result.get("filename", "")
        self.increment_progress(task_id, success=not is_error, from_cache=from_cache, last_item=filename)

    def list_tasks(self) -> List[dict]:
        with self._lock:
            return list(self._tasks.values())

    def delete_task(self, task_id: str) -> bool:
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
            return False

    def submit(self, fn, *args, **kwargs):
        return self._executor.submit(fn, *args, **kwargs)


task_manager = TaskManager()

app = FastAPI(
    title="Weather & Crowd Counting API",
    description="Get interpolated weather data and crowd counts for Balearic Islands",
    version="2.0.0"
)


# ============================================================
# WEATHER ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {"message": "Weather & Crowd Counting API", "docs": "/docs"}


@app.get("/weather")
def weather(
        lat: float = Query(..., ge=38.0, le=41.0),
        lon: float = Query(..., ge=0.5, le=5.0),
        datetime_str: str = Query(..., alias="datetime"),
        use_cache: bool = Query(True),
        time_window: int = Query(1, ge=1, le=12)
):
    try:
        target_dt = datetime.fromisoformat(datetime_str.replace(" ", "T"))
    except:
        raise HTTPException(400, "Invalid datetime format")

    result = get_weather_at_point(lat, lon, target_dt, use_cache=use_cache, time_window_hours=time_window)
    if not result:
        raise HTTPException(404, "No weather data available")

    return {"lat": lat, "lon": lon, "datetime": target_dt.isoformat(), "data": result}


@app.get("/weather/day")
def weather_day(
        lat: float = Query(...),
        lon: float = Query(...),
        date_str: str = Query(..., alias="date"),
        every_minute: bool = Query(False),
        name: Optional[str] = Query(None),
        use_cache: bool = Query(True)
):
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        raise HTTPException(400, "Invalid date format")

    results = []
    for hour in range(24):
        if every_minute:
            for minute in range(60):
                target_dt = datetime.combine(target_date, datetime.min.time().replace(hour=hour, minute=minute))
                data = get_weather_at_point(lat, lon, target_dt, use_cache=use_cache)
                results.append({"hour": hour, "minute": minute, "datetime": target_dt.isoformat(), "data": data})
        else:
            target_dt = datetime.combine(target_date, datetime.min.time().replace(hour=hour))
            data = get_weather_at_point(lat, lon, target_dt, use_cache=use_cache)
            results.append({"hour": hour, "datetime": target_dt.isoformat(), "data": data})

    return {"lat": lat, "lon": lon, "date": date_str, "name": name, "results": results}


class Location(BaseModel):
    lat: float
    lon: float
    datetime: str
    name: Optional[str] = None


class BatchRequest(BaseModel):
    locations: List[Location]
    use_cache: bool = True
    time_window: int = 1


@app.post("/weather/batch")
def weather_batch_post(request: BatchRequest):
    results = []
    for loc in request.locations:
        try:
            target_dt = datetime.fromisoformat(loc.datetime.replace(" ", "T"))
            data = get_weather_at_point(loc.lat, loc.lon, target_dt, use_cache=request.use_cache, time_window_hours=request.time_window)
            results.append({"lat": loc.lat, "lon": loc.lon, "datetime": target_dt.isoformat(), "name": loc.name, "data": data})
        except Exception as e:
            results.append({"lat": loc.lat, "lon": loc.lon, "datetime": loc.datetime, "name": loc.name, "error": str(e)})
    return {"count": len(results), "results": results}


# ============================================================
# PREDICTION ENDPOINTS
# ============================================================

@app.post("/predict")
async def predict_single(
        file: UploadFile = File(...),
        lat: float = Query(...),
        lon: float = Query(...),
        name: Optional[str] = Query(None, description="Custom name (e.g. 'Palma Beach', 'Camera 1')"),
        datetime_str: Optional[str] = Query(None, alias="datetime"),
        model_name: Optional[str] = Query(None, alias="model", description="Model to use for prediction"),
        use_cache: bool = Query(True)
):
    filename = file.filename
    used_model = model_name or get_model_info()["active"]

    # Check cache first
    if use_cache:
        cached = load_prediction_from_cache(filename, lat, lon, model=used_model, name=name)
        if cached:
            cached["from_cache"] = True
            return cached

    image_data = await file.read()

    if datetime_str:
        try:
            target_dt = datetime.fromisoformat(datetime_str.replace(" ", "T"))
        except:
            raise HTTPException(400, "Invalid datetime format")
    else:
        target_dt = parse_datetime_from_filename(filename)
        if target_dt is None:
            raise HTTPException(400, "Could not parse datetime from filename. Provide datetime parameter.")

    # Use specified model or active model
    if model_name:
        try:
            predictor = get_predictor(model_name)
            count = predictor.predict(image_data)
        except ValueError as e:
            raise HTTPException(404, str(e))
    else:
        count = predict_count(image_data)

    # Cleanup memory after prediction to prevent segfault
    cleanup_memory()

    weather_data = get_weather_at_point(lat, lon, target_dt, use_cache=True)

    result = {
        "filename": filename,
        "name": name,
        "datetime": target_dt.isoformat(),
        "lat": lat,
        "lon": lon,
        "count": count,
        "model": used_model,
        "weather": weather_data
    }

    if use_cache:
        save_prediction_to_cache(filename, lat, lon, result, model=used_model, name=name)

    return result


@app.post("/predict/batch")
async def predict_batch(
        files: List[UploadFile] = File(...),
        lat: float = Query(...),
        lon: float = Query(...),
        name: Optional[str] = Query(None, description="Custom name (e.g. 'Palma Beach', 'Camera 1')"),
        datetime_str: Optional[str] = Query(None, alias="datetime", description="Datetime for all images (if not parsing from filename)"),
        model_name: Optional[str] = Query(None, alias="model", description="Model to use for prediction"),
        use_cache: bool = Query(True),
        output_format: str = Query("json")
):
    results = []

    # Parse datetime if provided
    batch_datetime = None
    if datetime_str:
        try:
            batch_datetime = datetime.fromisoformat(datetime_str.replace(" ", "T"))
        except:
            raise HTTPException(400, "Invalid datetime format")

    # Get predictor once for batch
    if model_name:
        try:
            predictor = get_predictor(model_name)
        except ValueError as e:
            raise HTTPException(404, str(e))
    else:
        predictor = None

    used_model = model_name or get_model_info()["active"]

    for idx, file in enumerate(files):
        try:
            filename = file.filename

            # Check cache first
            if use_cache:
                cached = load_prediction_from_cache(filename, lat, lon, model=used_model, name=name)
                if cached:
                    cached["from_cache"] = True
                    results.append(cached)
                    continue

            image_data = await file.read()

            # Use batch datetime or parse from filename
            if batch_datetime:
                target_dt = batch_datetime
            else:
                target_dt = parse_datetime_from_filename(filename)
                if target_dt is None:
                    results.append({
                        "filename": filename,
                        "name": name,
                        "error": "Could not parse datetime from filename. Provide datetime parameter."
                    })
                    continue

            count = predictor.predict(image_data) if predictor else predict_count(image_data)

            # Cleanup memory after each prediction
            if idx % 5 == 0:  # Cleanup every 5 predictions to balance performance
                cleanup_memory()

            weather_data = get_weather_at_point(lat, lon, target_dt, use_cache=True)

            row = {
                "filename": filename,
                "name": name,
                "datetime": target_dt.isoformat(),
                "lat": lat,
                "lon": lon,
                "count": count,
                "model": used_model,
                "weather": weather_data
            }
            results.append(row)

            if use_cache:
                save_prediction_to_cache(filename, lat, lon, row, model=used_model, name=name)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "name": name,
                "error": str(e)
            })

    success_count = len([r for r in results if "error" not in r])
    error_count = len([r for r in results if "error" in r])

    # Final cleanup after batch
    cleanup_memory()

    if output_format == "csv":
        return _to_csv_response(results, "predictions.csv")

    return {"count": len(results), "success": success_count, "errors": error_count, "results": results}


class ImagePath(BaseModel):
    path: str
    lat: float
    lon: float
    name: Optional[str] = None
    datetime: Optional[str] = None


class ProcessPathsRequest(BaseModel):
    images: List[ImagePath]
    model: Optional[str] = None
    use_cache: bool = True
    output_format: str = "json"


@app.post("/predict/paths")
def predict_from_paths(request: ProcessPathsRequest):
    results = []

    # Get predictor once for batch
    if request.model:
        try:
            predictor = get_predictor(request.model)
        except ValueError as e:
            raise HTTPException(404, str(e))
    else:
        predictor = None

    used_model = request.model or get_model_info()["active"]

    for idx, img in enumerate(request.images):
        try:
            path = Path(img.path)
            filename = path.name

            # Check cache first
            if request.use_cache:
                cached = load_prediction_from_cache(filename, img.lat, img.lon, model=used_model, name=img.name)
                if cached:
                    cached["from_cache"] = True
                    results.append(cached)
                    continue

            if not path.exists():
                results.append({
                    "path": img.path,
                    "filename": filename,
                    "name": img.name,
                    "error": "File not found"
                })
                continue

            if img.datetime:
                target_dt = datetime.fromisoformat(img.datetime.replace(" ", "T"))
            else:
                target_dt = parse_datetime_from_filename(filename)
                if target_dt is None:
                    results.append({
                        "path": img.path,
                        "filename": filename,
                        "name": img.name,
                        "error": "Could not parse datetime"
                    })
                    continue

            count = predictor.predict(path) if predictor else predict_count(path)

            # Cleanup memory periodically
            if idx % 5 == 0:
                cleanup_memory()

            weather_data = get_weather_at_point(img.lat, img.lon, target_dt, use_cache=True)

            row = {
                "path": img.path,
                "filename": filename,
                "name": img.name,
                "datetime": target_dt.isoformat(),
                "lat": img.lat,
                "lon": img.lon,
                "count": count,
                "model": used_model,
                "weather": weather_data
            }
            results.append(row)

            if request.use_cache:
                save_prediction_to_cache(filename, img.lat, img.lon, row, model=used_model, name=img.name)
        except Exception as e:
            results.append({
                "path": img.path,
                "filename": Path(img.path).name if img.path else None,
                "name": img.name,
                "error": str(e)
            })

    success_count = len([r for r in results if "error" not in r])
    error_count = len([r for r in results if "error" in r])

    # Final cleanup
    cleanup_memory()

    if request.output_format == "csv":
        return _to_csv_response(results, "predictions.csv")

    return {"count": len(results), "success": success_count, "errors": error_count, "results": results}


@app.post("/predict/directory")
def predict_directory(
        directory: str = Query(..., description="Source directory containing images"),
        lat: float = Query(...),
        lon: float = Query(...),
        cache_name: str = Query('', description="Cache folder name to create"),
        model_name: Optional[str] = Query(None, alias="model", description="Model to use for prediction"),
        pattern: str = Query("*.jpg"),
        recursive: bool = Query(True, description="Scan subfolders"),
        use_cache: bool = Query(True),
        background: bool = Query(False, description="Run as background task"),
        limit: Optional[int] = Query(None, description="Limit number of images to process"),
        output_format: str = Query("json")
):
    """Process all images in a directory. Set background=true to run async."""
    import glob

    dir_path = Path(directory)
    if not dir_path.exists():
        raise HTTPException(404, "Directory not found")

    if recursive:
        image_files = sorted(glob.glob(str(dir_path / "**" / pattern), recursive=True))
    else:
        image_files = sorted(glob.glob(str(dir_path / pattern)))

    if not image_files:
        raise HTTPException(404, f"No files matching {pattern}")

    # Apply limit
    total_found = len(image_files)
    if limit:
        image_files = image_files[:limit]

    used_model = model_name or get_model_info()["active"]

    # Validate model exists
    if model_name:
        try:
            get_predictor(model_name)
        except ValueError as e:
            raise HTTPException(404, str(e))

    params = {
        "directory": directory,
        "lat": lat,
        "lon": lon,
        "cache_name": cache_name,
        "model": used_model,
        "pattern": pattern,
        "recursive": recursive,
        "use_cache": use_cache,
        "total_files": len(image_files),
        "total_found": total_found,
        "limit": limit
    }

    if background:
        task_id = task_manager.create_task("directory_prediction", params)
        task_manager.update_task(task_id, total=len(image_files), status="running", started_at=datetime.now().isoformat())

        task_manager.submit(
            _process_directory_task,
            task_id, image_files, dir_path, lat, lon, cache_name, used_model, use_cache
        )

        response = {
            "task_id": task_id,
            "status": "running",
            "total": len(image_files),
            "message": f"Processing {len(image_files)} images in background",
            "check_status": f"/tasks/{task_id}"
        }
        if limit:
            response["limit"] = limit
            response["total_found"] = total_found
        return response

    # Synchronous processing
    return _process_directory_sync(image_files, dir_path, lat, lon, cache_name, used_model, use_cache, output_format)


def _process_directory_task(task_id: str, image_files: list, dir_path: Path, lat: float, lon: float,
                            cache_name: str, used_model: str, use_cache: bool):
    """Background task for directory processing - memory efficient."""
    import gc

    try:
        predictor = get_predictor(used_model)
        total = len(image_files)

        for idx, img_path in enumerate(image_files):
            try:
                path = Path(img_path)
                filename = path.name
                rel_path = path.parent.relative_to(dir_path)
                subpath = str(rel_path) if str(rel_path) != "." else None

                if use_cache:
                    cached = load_prediction_from_cache(filename, lat, lon, model=used_model, name=cache_name, subpath=subpath)
                    if cached:
                        task_manager.increment_progress(task_id, success=True, from_cache=True, last_item=filename)
                        del cached
                        continue

                if not path.exists():
                    task_manager.increment_progress(task_id, success=False, last_item=filename)
                    continue

                target_dt = parse_datetime_from_filename(filename)
                if target_dt is None:
                    task_manager.increment_progress(task_id, success=False, last_item=filename)
                    continue

                count = predictor.predict(path)
                weather_data = get_weather_at_point(lat, lon, target_dt, use_cache=True)

                row = {
                    "path": img_path,
                    "filename": filename,
                    "subpath": subpath,
                    "cache_name": cache_name,
                    "datetime": target_dt.isoformat(),
                    "lat": lat,
                    "lon": lon,
                    "count": count,
                    "model": used_model,
                    "weather": weather_data
                }

                if use_cache:
                    save_prediction_to_cache(filename, lat, lon, row, model=used_model, name=cache_name, subpath=subpath)

                task_manager.increment_progress(task_id, success=True, from_cache=False, last_item=filename)
                del row, weather_data, count

                # AGGRESSIVE cleanup every 5 images
                if idx % 5 == 0:
                    gc.collect()
                    cleanup_memory()

                if idx % 50 == 0 and idx > 0:
                    print(f"[Task {task_id}] Processed {idx}/{total} ({idx * 100 // total}%)")

            except Exception as e:
                task_manager.increment_progress(task_id, success=False, last_item=Path(img_path).name if img_path else None)

        gc.collect()
        cleanup_memory()
        task_manager.update_task(task_id, status="completed", completed_at=datetime.now().isoformat())
        print(f"[Task {task_id}] Completed!")

    except Exception as e:
        task_manager.update_task(task_id, status="failed", error=str(e), completed_at=datetime.now().isoformat())


def _process_directory_sync(image_files: list, dir_path: Path, lat: float, lon: float,
                            cache_name: str, used_model: str, use_cache: bool, output_format: str):
    """Synchronous directory processing."""
    predictor = get_predictor(used_model)
    results = []

    for idx, img_path in enumerate(image_files):
        try:
            path = Path(img_path)
            filename = path.name
            rel_path = path.parent.relative_to(dir_path)
            subpath = str(rel_path) if str(rel_path) != "." else None

            if use_cache:
                cached = load_prediction_from_cache(filename, lat, lon, model=used_model, name=cache_name, subpath=subpath)
                if cached:
                    cached["from_cache"] = True
                    results.append(cached)
                    continue

            if not path.exists():
                results.append({"path": img_path, "filename": filename, "subpath": subpath, "cache_name": cache_name, "error": "File not found"})
                continue

            target_dt = parse_datetime_from_filename(filename)
            if target_dt is None:
                results.append({"path": img_path, "filename": filename, "subpath": subpath, "cache_name": cache_name, "error": "Could not parse datetime"})
                continue

            count = predictor.predict(path)

            # Cleanup memory periodically
            if idx % 5 == 0:
                cleanup_memory()

            weather_data = get_weather_at_point(lat, lon, target_dt, use_cache=True)

            row = {
                "path": img_path, "filename": filename, "subpath": subpath, "cache_name": cache_name,
                "datetime": target_dt.isoformat(), "lat": lat, "lon": lon,
                "count": count, "model": used_model, "weather": weather_data
            }
            results.append(row)

            if use_cache:
                save_prediction_to_cache(filename, lat, lon, row, model=used_model, name=cache_name, subpath=subpath)
        except Exception as e:
            results.append({"path": img_path, "filename": Path(img_path).name if img_path else None, "cache_name": cache_name, "error": str(e)})

    success_count = len([r for r in results if "error" not in r])
    error_count = len([r for r in results if "error" in r])

    # Final cleanup
    cleanup_memory()

    if output_format == "csv":
        return _to_csv_response(results, f"{cache_name}_predictions.csv")

    return {"directory": str(dir_path), "cache_name": cache_name, "count": len(results), "success": success_count, "errors": error_count, "results": results}


# ============================================================
# TASK ENDPOINTS
# ============================================================

@app.get("/tasks")
def list_tasks():
    """List all tasks."""
    return {"tasks": task_manager.list_tasks()}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """Get task status and progress."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    return {
        "id": task["id"],
        "type": task["type"],
        "status": task["status"],
        "progress": task["progress"],
        "current": task["current"],
        "total": task["total"],
        "success": task["success"],
        "errors": task["errors"],
        "from_cache": task.get("from_cache", 0),
        "last_processed": task.get("last_processed"),
        "params": task["params"],
        "created_at": task["created_at"],
        "started_at": task["started_at"],
        "completed_at": task["completed_at"],
        "error": task["error"],
        "note": "Results are saved to cache. Use /cache/list to see cached predictions."
    }


@app.get("/tasks/{task_id}/results")
def get_task_results(task_id: str):
    """Task results are now stored in cache, not in memory."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    return {
        "task_id": task_id,
        "status": task["status"],
        "success": task["success"],
        "errors": task["errors"],
        "from_cache": task.get("from_cache", 0),
        "message": "Results are stored in the prediction cache. Use /cache/list or load from cache directory.",
        "cache_hint": f"Look in cache/predictions/ for the processed files"
    }


@app.delete("/tasks/{task_id}")
def delete_task(task_id: str):
    """Delete a task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    if task["status"] == "running":
        raise HTTPException(400, "Cannot delete running task")

    task_manager.delete_task(task_id)
    return {"message": f"Task {task_id} deleted"}


# ============================================================
# MULTI-BEACH PROCESSING
# ============================================================

class BeachCoordinates(BaseModel):
    lat: float
    lon: float
    name: Optional[str] = None  # Display name (defaults to folder name)


class BeachesRequest(BaseModel):
    directory: str
    beaches: Optional[dict[str, BeachCoordinates]] = None  # folder_name -> coordinates (or use beaches.json)
    model: Optional[str] = None
    pattern: str = "*.jpg"
    use_cache: bool = True
    background: bool = False
    output_format: str = "json"
    limit_per_folder: Optional[int] = None  # Limit images per folder (for testing)


class TrainRequest(BaseModel):
    data_source: str  # Path to cache dir or JSON file
    model_type: str = "random_forest"  # ML model: random_forest, ridge, xgboost (if installed)
    counting_model: Optional[str] = None  # Filter by counting model (e.g., bayesian_vgg19)
    beach_filter: Optional[str] = None  # Filter by beach
    include_weather: bool = True
    include_temporal: bool = True
    include_location: bool = False
    custom_features: Optional[List[str]] = None
    test_size: float = 0.2
    time_series_split: bool = True
    save_name: Optional[str] = None
    save_dir: Optional[str] = None


class CompareRequest(BaseModel):
    data_source: str
    counting_model: Optional[str] = None
    beach_filter: Optional[str] = None
    include_weather: bool = True
    include_temporal: bool = True
    include_location: bool = False
    models: Optional[List[str]] = None  # List of models to compare (None = all)
    cv_folds: int = 5
    time_series_cv: bool = False


@app.get("/predict/beaches/scan")
def scan_beaches_directory(
        directory: str = Query(..., description="Directory to scan"),
        pattern: str = Query("*.jpg")
):
    """
    Scan directory and return all folders containing images with their paths.
    Use this to see what folders need coordinates before calling /predict/beaches.
    """
    import glob

    dir_path = Path(directory)
    if not dir_path.exists():
        raise HTTPException(404, "Directory not found")

    # Find all images recursively
    all_images = glob.glob(str(dir_path / "**" / pattern), recursive=True)

    if not all_images:
        raise HTTPException(404, f"No images matching {pattern}")

    # Group by relative folder path
    folders = {}
    for img_path in all_images:
        path = Path(img_path)
        rel_folder = str(path.parent.relative_to(dir_path))

        if rel_folder not in folders:
            folders[rel_folder] = {"images": 0, "path": str(path.parent)}
        folders[rel_folder]["images"] += 1

    # Sort folders
    folders = dict(sorted(folders.items()))
    total_images = sum(f["images"] for f in folders.values())

    return {
        "directory": directory,
        "total_folders": len(folders),
        "total_images": total_images,
        "folders": folders,
        "template": {
            folder_path: {"lat": 0.0, "lon": 0.0, "name": folder_path.split("/")[-1]}
            for folder_path in folders.keys()
        }
    }


class SaveConfigRequest(BaseModel):
    directory: str
    beaches: dict[str, BeachCoordinates]


@app.post("/predict/beaches/config")
def save_beaches_config(request: SaveConfigRequest):
    """
    Save beaches configuration to beaches.json in the directory.
    This file will be automatically loaded when calling /predict/beaches without beaches parameter.
    """
    dir_path = Path(request.directory)
    if not dir_path.exists():
        raise HTTPException(404, "Directory not found")

    config_file = dir_path / "beaches.json"

    try:
        config_data = {k: v.model_dump() for k, v in request.beaches.items()}
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        return {
            "message": "Configuration saved",
            "file": str(config_file),
            "beaches": len(request.beaches)
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to save config: {e}")


@app.get("/predict/beaches/config")
def get_beaches_config(directory: str = Query(...)):
    """Get beaches.json configuration from a directory."""
    dir_path = Path(directory)
    config_file = dir_path / "beaches.json"

    if not config_file.exists():
        raise HTTPException(404, "No beaches.json found in directory")

    try:
        with open(config_file, 'r') as f:
            data = json.load(f)
        return {"file": str(config_file), "beaches": data}
    except Exception as e:
        raise HTTPException(500, f"Failed to read config: {e}")


@app.post("/predict/beaches")
def predict_beaches(request: BeachesRequest):
    """
    Process multiple beach folders, each with its own coordinates.
    Supports nested paths like "skyline/cala-vadella-ibiza".
    Folders without coordinates are skipped (not blocking).

    Example request:
    {
        "directory": "/images",
        "beaches": {
            "skyline/cala-vadella-ibiza": {"lat": 38.91, "lon": 1.22, "name": "Cala Vedella"},
            "skyline/es-pujols": {"lat": 38.73, "lon": 1.46, "name": "Es Pujols"}
        }
    }

    Or use a beaches.json config file in the directory.
    """
    import glob

    dir_path = Path(request.directory)
    if not dir_path.exists():
        raise HTTPException(404, "Directory not found")

    # Load beaches from config file if not provided
    beaches_config = request.beaches
    if not beaches_config:
        config_file = dir_path / "beaches.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                beaches_config = {k: BeachCoordinates(**v) for k, v in data.items()}
            except Exception as e:
                raise HTTPException(400, f"Error reading beaches.json: {e}")
        else:
            raise HTTPException(400, "No beaches provided and no beaches.json found in directory")

    # Find all images recursively
    all_image_files = glob.glob(str(dir_path / "**" / request.pattern), recursive=True)

    if not all_image_files:
        raise HTTPException(404, f"No images matching {request.pattern}")

    # Group images by their parent folder's relative path
    folder_images = {}
    for img_path in all_image_files:
        path = Path(img_path)
        rel_folder = str(path.parent.relative_to(dir_path))

        if rel_folder not in folder_images:
            folder_images[rel_folder] = []
        folder_images[rel_folder].append(img_path)

    # Separate folders with and without coordinates
    skipped_folders = []
    matched_folders = []
    for folder_path in folder_images.keys():
        if folder_path in beaches_config:
            matched_folders.append(folder_path)
        else:
            skipped_folders.append(folder_path)

    if not matched_folders:
        raise HTTPException(400, {
            "error": "No folders match the provided coordinates",
            "available_folders": sorted(folder_images.keys()),
            "provided_coordinates": sorted(beaches_config.keys()),
            "hint": "Use /predict/beaches/scan to see all folders"
        })

    # Validate model
    used_model = request.model or get_model_info()["active"]
    if request.model:
        try:
            get_predictor(request.model)
        except ValueError as e:
            raise HTTPException(404, str(e))

    # Collect images only from folders with coordinates
    all_images = []
    for folder_path in matched_folders:
        beach_config = beaches_config[folder_path]
        folder_imgs = sorted(folder_images[folder_path])

        # Apply limit if specified
        if request.limit_per_folder:
            folder_imgs = folder_imgs[:request.limit_per_folder]

        for img_path in folder_imgs:
            path = Path(img_path)

            all_images.append({
                "path": img_path,
                "filename": path.name,
                "beach_folder": folder_path,
                "beach_name": beach_config.name or folder_path,
                "lat": beach_config.lat,
                "lon": beach_config.lon
            })

    params = {
        "directory": request.directory,
        "beaches": {k: v.model_dump() for k, v in beaches_config.items() if k in matched_folders},
        "model": used_model,
        "pattern": request.pattern,
        "use_cache": request.use_cache,
        "total_files": len(all_images),
        "total_beaches": len(matched_folders),
        "skipped_folders": skipped_folders,
        "limit_per_folder": request.limit_per_folder
    }

    if request.background:
        task_id = task_manager.create_task("beaches_prediction", params)
        task_manager.update_task(task_id, total=len(all_images), status="running", started_at=datetime.now().isoformat())

        task_manager.submit(
            _process_beaches_task,
            task_id, all_images, used_model, request.use_cache
        )

        response = {
            "task_id": task_id,
            "status": "running",
            "total_images": len(all_images),
            "total_beaches": len(matched_folders),
            "beaches": sorted(matched_folders),
            "message": f"Processing {len(all_images)} images from {len(matched_folders)} beaches",
            "check_status": f"/tasks/{task_id}"
        }

        if request.limit_per_folder:
            response["limit_per_folder"] = request.limit_per_folder

        if skipped_folders:
            response["skipped_folders"] = sorted(skipped_folders)
            response["skipped_count"] = len(skipped_folders)

        return response

    # Synchronous processing
    result = _process_beaches_sync(all_images, used_model, request.use_cache, request.output_format)

    if isinstance(result, dict):
        if request.limit_per_folder:
            result["limit_per_folder"] = request.limit_per_folder
        if skipped_folders:
            result["skipped_folders"] = sorted(skipped_folders)
            result["skipped_count"] = len(skipped_folders)

    return result


def _process_beaches_task(task_id: str, images: list, used_model: str, use_cache: bool):
    """Background task for multi-beach processing - memory efficient."""
    import gc

    try:
        predictor = get_predictor(used_model)
        total = len(images)

        for idx, img_info in enumerate(images):
            try:
                path = Path(img_info["path"])
                filename = img_info["filename"]
                lat = img_info["lat"]
                lon = img_info["lon"]
                beach_folder = img_info["beach_folder"]
                beach_name = img_info["beach_name"]

                # Cache uses beach_folder (full path like "skyline/cala-vadella-ibiza")
                if use_cache:
                    cached = load_prediction_from_cache(filename, lat, lon, model=used_model, name=beach_folder)
                    if cached:
                        task_manager.increment_progress(task_id, success=True, from_cache=True, last_item=filename)
                        del cached  # Release memory
                        continue

                if not path.exists():
                    task_manager.increment_progress(task_id, success=False, last_item=filename)
                    continue

                target_dt = parse_datetime_from_filename(filename)
                if target_dt is None:
                    task_manager.increment_progress(task_id, success=False, last_item=filename)
                    continue

                count = predictor.predict(path)
                weather_data = get_weather_at_point(lat, lon, target_dt, use_cache=True)

                row = {
                    "path": str(path),
                    "filename": filename,
                    "beach": beach_name,
                    "beach_folder": beach_folder,
                    "datetime": target_dt.isoformat(),
                    "lat": lat,
                    "lon": lon,
                    "count": count,
                    "model": used_model,
                    "weather": weather_data
                }

                if use_cache:
                    save_prediction_to_cache(filename, lat, lon, row, model=used_model, name=beach_folder)

                task_manager.increment_progress(task_id, success=True, from_cache=False, last_item=filename)

                # Release references immediately
                del row, weather_data, count

                # AGGRESSIVE memory cleanup every 5 images
                if idx % 5 == 0:
                    gc.collect()
                    cleanup_memory()

                # Log progress
                if idx % 50 == 0 and idx > 0:
                    print(f"[Task {task_id}] Processed {idx}/{total} ({idx * 100 // total}%)")

            except Exception as e:
                task_manager.increment_progress(task_id, success=False, last_item=img_info.get("filename"))
                print(f"[Task {task_id}] Error processing {img_info.get('filename')}: {e}")

        # Final cleanup
        gc.collect()
        cleanup_memory()
        task_manager.update_task(task_id, status="completed", completed_at=datetime.now().isoformat())
        print(f"[Task {task_id}] Completed!")

    except Exception as e:
        task_manager.update_task(task_id, status="failed", error=str(e), completed_at=datetime.now().isoformat())


def _process_beaches_sync(images: list, used_model: str, use_cache: bool, output_format: str):
    """Synchronous multi-beach processing."""
    predictor = get_predictor(used_model)
    results = []
    beaches_summary = {}

    for idx, img_info in enumerate(images):
        try:
            path = Path(img_info["path"])
            filename = img_info["filename"]
            lat = img_info["lat"]
            lon = img_info["lon"]
            beach_folder = img_info["beach_folder"]
            beach_name = img_info["beach_name"]

            if beach_name not in beaches_summary:
                beaches_summary[beach_name] = {"total": 0, "success": 0, "errors": 0}
            beaches_summary[beach_name]["total"] += 1

            if use_cache:
                cached = load_prediction_from_cache(filename, lat, lon, model=used_model, name=beach_folder)
                if cached:
                    cached["from_cache"] = True
                    cached["beach"] = beach_name
                    results.append(cached)
                    beaches_summary[beach_name]["success"] += 1
                    continue

            if not path.exists():
                results.append({"path": str(path), "filename": filename, "beach": beach_name, "beach_folder": beach_folder, "error": "File not found"})
                beaches_summary[beach_name]["errors"] += 1
                continue

            target_dt = parse_datetime_from_filename(filename)
            if target_dt is None:
                results.append({"path": str(path), "filename": filename, "beach": beach_name, "beach_folder": beach_folder, "error": "Could not parse datetime"})
                beaches_summary[beach_name]["errors"] += 1
                continue

            count = predictor.predict(path)

            # Cleanup memory periodically to prevent segfault
            if idx % 5 == 0:
                cleanup_memory()

            weather_data = get_weather_at_point(lat, lon, target_dt, use_cache=True)

            row = {
                "path": str(path), "filename": filename, "beach": beach_name,
                "beach_folder": beach_folder,
                "datetime": target_dt.isoformat(), "lat": lat, "lon": lon,
                "count": count, "model": used_model, "weather": weather_data
            }
            results.append(row)
            beaches_summary[beach_name]["success"] += 1

            if use_cache:
                save_prediction_to_cache(filename, lat, lon, row, model=used_model, name=beach_folder)

        except Exception as e:
            results.append({"path": img_info.get("path"), "filename": img_info.get("filename"), "beach": img_info.get("beach_name"), "beach_folder": img_info.get("beach_folder"),
                            "error": str(e)})
            if img_info.get("beach_name") in beaches_summary:
                beaches_summary[img_info["beach_name"]]["errors"] += 1

    success_count = len([r for r in results if "error" not in r])
    error_count = len([r for r in results if "error" in r])

    # Final cleanup
    cleanup_memory()

    if output_format == "csv":
        return _to_csv_response(results, "beaches_predictions.csv")

    return {
        "count": len(results),
        "success": success_count,
        "errors": error_count,
        "beaches": beaches_summary,
        "results": results
    }


# ============================================================
# UTILITY
# ============================================================

def _to_csv_response(results, filename):
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"count": len(results), "results": results}

    # Flatten weather dict for CSV
    flat_results = []
    for r in valid:
        row = {k: v for k, v in r.items() if k != "weather" and k != "from_cache"}
        if "weather" in r and r["weather"]:
            row.update(r["weather"])
        flat_results.append(row)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(flat_results[0].keys()))
    writer.writeheader()
    writer.writerows(flat_results)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/info")
def info():
    station_info = get_station_info()
    model_info = get_model_info()
    return {
        "weather": {
            "records": station_info.get("records", 0),
            "stations": station_info.get("stations", 0),
            "years": station_info.get("years", []),
            "alt_threshold": station_info.get("alt_threshold", 300),
            "date_range": {
                "start": str(station_info.get("date_range", (None, None))[0]),
                "end": str(station_info.get("date_range", (None, None))[1])
            } if station_info.get("date_range") else None
        },
        "model": model_info
    }


@app.get("/health")
def health():
    """Check API health including external service connectivity."""
    station_info = get_station_info()
    openmeteo_status = check_openmeteo_connectivity()
    return {
        "status": "ok",
        "aemet": {
            "status": "ok" if station_info.get("records", 0) > 0 else "no_data",
            "records": station_info.get("records", 0)
        },
        "openmeteo": openmeteo_status
    }


@app.post("/reload")
def reload():
    reload_aemet_data()
    return {"message": "AEMET data reloaded"}


@app.delete("/cache/predictions")
def clear_cache(
        model: Optional[str] = Query(None, description="Clear specific model's cache"),
        name: Optional[str] = Query(None, description="Clear specific name's cache")
):
    """Clear prediction cache files. Can filter by model and/or name."""
    count = clear_prediction_cache(model=model, name=name)
    return {"message": f"Cleared {count} cache files", "model": model, "name": name}


@app.get("/cache/predictions")
def list_caches():
    """List all prediction cache directories and file counts."""
    return {"caches": list_cache_names()}


@app.post("/load-model")
def load_model_endpoint(name: Optional[str] = Query(None, description="Model name to load (uses active if not specified)")):
    """Load a prediction model."""
    try:
        if name:
            predictor = get_predictor(name)
            predictor.load()
        else:
            load_model()
        return {"message": "Model loaded", "info": get_model_info()}
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {str(e)}")


@app.post("/model/switch")
def switch_model(name: str = Query(..., description="Model name to switch to")):
    """Switch active prediction model."""
    try:
        set_active_predictor(name)
        return {"message": f"Switched to model: {name}", "info": get_model_info()}
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/model")
def get_models_endpoint():
    """Get all registered models and active model."""
    return get_model_info()


# ============================================================
# TRAINING ENDPOINTS
# ============================================================

@app.get("/train/models")
def list_ml_models():
    """List available ML models for training."""
    if not HAS_TRAINING:
        raise HTTPException(500, "Training module not available")

    return {
        "models": list_available_models(),
        "description": {
            "xgboost": "Gradient boosting (XGBoost) - fast, good default",
            "lightgbm": "Light Gradient Boosting - fast, handles large data",
            "random_forest": "Random Forest - robust, interpretable",
            "gradient_boosting": "Sklearn Gradient Boosting - slower but stable",
            "ridge": "Ridge Regression - linear, fast, good baseline",
            "lasso": "Lasso Regression - linear with feature selection",
            "elastic_net": "Elastic Net - combines Ridge and Lasso",
            "svr": "Support Vector Regression - good for small data"
        }
    }


@app.get("/train/data/summary")
def get_data_summary(
        cache_dir: str = Query(..., description="Path to cache directory"),
        counting_model: Optional[str] = Query(None, description="Filter by counting model")
):
    """Get summary of available training data in cache."""
    if not HAS_TRAINING:
        raise HTTPException(500, "Training module not available")

    try:
        summary = summarize_cache(cache_dir)

        if counting_model and len(summary) > 0:
            summary = summary[summary['model'] == counting_model]

        return {
            "cache_dir": cache_dir,
            "counting_model_filter": counting_model,
            "total_records": int(summary['count'].sum()) if len(summary) > 0 else 0,
            "beaches": len(summary),
            "data": summary.to_dict('records') if len(summary) > 0 else []
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/train/data/preview")
def preview_training_data(
        cache_dir: str = Query(..., description="Path to cache directory"),
        counting_model: Optional[str] = Query(None),
        beach: Optional[str] = Query(None),
        limit: int = Query(10, ge=1, le=100)
):
    """Preview training data from cache."""
    if not HAS_TRAINING:
        raise HTTPException(500, "Training module not available")

    try:
        df = load_training_data(cache_dir, model=counting_model, beach=beach)
        df = add_temporal_features(df)

        # Get feature info
        weather_cols = [c for c in df.columns if c.startswith('ae_') or c.startswith('om_')]
        temporal_cols = ['hour', 'day_of_week', 'month', 'is_weekend']

        preview = df.head(limit).copy()
        preview['datetime'] = preview['datetime'].astype(str)

        return {
            "total_records": len(df),
            "beaches": df['beach_folder'].nunique() if 'beach_folder' in df.columns else 0,
            "date_range": {
                "min": str(df['datetime'].min()),
                "max": str(df['datetime'].max())
            } if 'datetime' in df.columns else None,
            "count_stats": {
                "min": float(df['count'].min()),
                "max": float(df['count'].max()),
                "mean": float(df['count'].mean()),
                "std": float(df['count'].std())
            } if 'count' in df.columns else None,
            "features": {
                "weather": weather_cols,
                "temporal": [c for c in temporal_cols if c in df.columns]
            },
            "preview": preview.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/train/compare")
def compare_ml_models(request: CompareRequest):
    """Compare multiple ML models using cross-validation."""
    if not HAS_TRAINING:
        raise HTTPException(500, "Training module not available")

    try:
        df = load_from_path(request.data_source, model=request.counting_model, beach=request.beach_filter)

        X, y, features = prepare_data(
            df,
            include_weather=request.include_weather,
            include_temporal=request.include_temporal,
            include_location=request.include_location
        )

        if len(X) < request.cv_folds * 2:
            raise HTTPException(400, f"Not enough data for {request.cv_folds}-fold CV. Have {len(X)} samples.")

        comparison = compare_models(
            X, y,
            models=request.models,
            cv_folds=request.cv_folds,
            time_series_cv=request.time_series_cv
        )

        return {
            "samples": len(X),
            "features": len(features),
            "feature_names": features,
            "cv_folds": request.cv_folds,
            "time_series_cv": request.time_series_cv,
            "results": comparison.to_dict('records')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


def _train_task(task_id: str, request_dict: dict):
    """Background training task."""
    try:
        task_manager.update_task(task_id, status="loading_data", progress=10)

        df = load_from_path(
            request_dict['data_source'],
            model=request_dict.get('counting_model'),
            beach=request_dict.get('beach_filter')
        )

        task_manager.update_task(task_id, status="preparing_features", progress=30)

        X, y, features = prepare_data(
            df,
            include_weather=request_dict.get('include_weather', True),
            include_temporal=request_dict.get('include_temporal', True),
            include_location=request_dict.get('include_location', False),
            custom_features=request_dict.get('custom_features')
        )

        task_manager.update_task(
            task_id,
            status="training",
            progress=50,
            total=len(X),
            current=0
        )

        results = train_model(
            X, y,
            model_type=request_dict.get('model_type', 'xgboost'),
            test_size=request_dict.get('test_size', 0.2),
            time_series_split=request_dict.get('time_series_split', True)
        )

        task_manager.update_task(task_id, status="saving", progress=90)

        # Save model if requested
        saved_path = None
        if request_dict.get('save_name'):
            saved_path = save_trained_model(
                results,
                request_dict['save_name'],
                save_dir=request_dict.get('save_dir')
            )

        # Prepare results for JSON
        train_results = {
            "model_type": results['model_type'],
            "features": results['features'],
            "train_size": results['train_size'],
            "test_size": results['test_size'],
            "metrics": results['metrics'],
            "feature_importance": results.get('feature_importance'),
            "saved_path": saved_path
        }

        task_manager.add_result(task_id, train_results)
        task_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            completed_at=datetime.now().isoformat()
        )

    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            completed_at=datetime.now().isoformat()
        )


@app.post("/train")
def start_training(request: TrainRequest, background: bool = Query(True)):
    """
    Train a prediction model.

    Set background=true (default) to run as background task.
    """
    if not HAS_TRAINING:
        raise HTTPException(500, "Training module not available")

    request_dict = request.model_dump()

    if background:
        task_id = task_manager.create_task("training", request_dict)
        task_manager.update_task(task_id, status="starting", started_at=datetime.now().isoformat())

        task_manager.submit(_train_task, task_id, request_dict)

        return {
            "task_id": task_id,
            "status": "starting",
            "message": f"Training {request.model_type} model in background",
            "check_status": f"/tasks/{task_id}",
            "get_results": f"/tasks/{task_id}/results"
        }

    # Synchronous training
    try:
        df = load_from_path(request.data_source, model=request.counting_model, beach=request.beach_filter)

        X, y, features = prepare_data(
            df,
            include_weather=request.include_weather,
            include_temporal=request.include_temporal,
            include_location=request.include_location,
            custom_features=request.custom_features
        )

        results = train_model(
            X, y,
            model_type=request.model_type,
            test_size=request.test_size,
            time_series_split=request.time_series_split
        )

        saved_path = None
        if request.save_name:
            saved_path = save_trained_model(results, request.save_name, save_dir=request.save_dir)

        return {
            "model_type": results['model_type'],
            "samples": len(X),
            "features": results['features'],
            "train_size": results['train_size'],
            "test_size": results['test_size'],
            "metrics": results['metrics'],
            "feature_importance": results.get('feature_importance'),
            "saved_path": saved_path
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/train/saved")
def list_saved_models(directory: Optional[str] = Query(None)):
    """List saved trained models."""
    if not HAS_TRAINING:
        raise HTTPException(500, "Training module not available")

    import pickle

    search_dir = Path(directory) if directory else Path.cwd() / 'models/prediction'

    if not search_dir.exists():
        return {"models": [], "directory": str(search_dir)}

    models = []
    for pkl_file in search_dir.rglob("*.pkl"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            models.append({
                'name': pkl_file.stem,
                'path': str(pkl_file),
                'model_type': data.get('model_type'),
                'features': len(data.get('features', [])),
                'metrics': data.get('metrics', {}).get('test'),
                'saved_at': data.get('saved_at')
            })
        except:
            pass

    return {"models": models, "directory": str(search_dir)}


@app.post("/train/predict")
def predict_with_trained_model(
        model_path: str = Query(..., description="Path to saved model .pkl file"),
        cache_dir: str = Query(..., description="Path to prediction cache to use for prediction"),
        counting_model: Optional[str] = Query(None),
        beach: Optional[str] = Query(None),
        limit: Optional[int] = Query(None)
):
    """Use a trained model to make predictions on data."""
    if not HAS_TRAINING:
        raise HTTPException(500, "Training module not available")

    try:
        model_data = load_trained_model(model_path)
        df = load_training_data(cache_dir, model=counting_model, beach=beach)

        if limit:
            df = df.head(limit)

        df = add_temporal_features(df)

        predictions = predict_with_model(model_data, df)

        df['predicted_count'] = predictions
        df['actual_count'] = df['count']
        df['error'] = df['predicted_count'] - df['actual_count']
        df['abs_error'] = df['error'].abs()

        # Metrics
        mae = df['abs_error'].mean()
        rmse = (df['error'] ** 2).mean() ** 0.5

        results = df[['filename', 'beach', 'datetime', 'actual_count', 'predicted_count', 'error']].copy()
        results['datetime'] = results['datetime'].astype(str)

        return {
            "model_path": model_path,
            "model_type": model_data.get('model_type'),
            "samples": len(df),
            "metrics": {
                "mae": float(mae),
                "rmse": float(rmse)
            },
            "predictions": results.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(400, str(e))


class ForecastRequest(BaseModel):
    model_path: str  # Path to trained model
    datetime: str  # ISO format datetime for prediction
    lat: float  # Beach latitude
    lon: float  # Beach longitude
    weather: Optional[dict] = None  # Manual weather data (if not provided, fetches forecast)
    beach_name: Optional[str] = None


class MultiForecastRequest(BaseModel):
    model_path: str
    lat: float
    lon: float
    start_datetime: str  # Start of forecast period
    hours: int = 24  # Number of hours to forecast
    weather_forecasts: Optional[List[dict]] = None  # Manual weather per hour
    beach_name: Optional[str] = None


@app.post("/forecast")
def forecast_crowd(request: ForecastRequest):
    """
    Predict crowd count for a future date/time using trained model.

    If weather data is not provided, it will try to fetch forecast from Open-Meteo.

    Example request:
    {
        "model_path": "models/prediction/crowd_xgboost.pkl",
        "datetime": "2024-01-29T14:00:00",
        "lat": 39.534686,
        "lon": 2.717979,
        "weather": {
            "ae_ta": 22.0,
            "om_temperature_2m": 21.5,
            "om_cloud_cover": 20.0,
            "om_wind_speed_10m": 10.0
        }
    }
    """
    if not HAS_TRAINING:
        raise HTTPException(500, "Training module not available")

    import pandas as pd
    import numpy as np

    try:
        # Load model
        model_data = load_trained_model(request.model_path)
        features = model_data['features']

        # Parse datetime
        target_dt = datetime.fromisoformat(request.datetime.replace(" ", "T"))

        # Build feature row
        row = {
            'datetime': target_dt,
            'lat': request.lat,
            'lon': request.lon
        }

        # Get weather data
        if request.weather:
            row.update(request.weather)
        else:
            # Try to fetch forecast from Open-Meteo
            try:
                weather = get_weather_at_point(request.lat, request.lon, target_dt, use_cache=False)
                if weather:
                    row.update(weather)
            except:
                pass

        # Create DataFrame and add temporal features
        df = pd.DataFrame([row])
        df = add_temporal_features(df)

        # Check which features are missing
        missing = [f for f in features if f not in df.columns]
        available = [f for f in features if f in df.columns]

        if missing:
            # Fill missing with 0 or mean (not ideal but allows prediction)
            for f in missing:
                df[f] = 0.0

        # Predict
        prediction = predict_with_model(model_data, df)
        predicted_count = max(0, float(prediction[0]))  # Can't have negative people

        return {
            "datetime": target_dt.isoformat(),
            "lat": request.lat,
            "lon": request.lon,
            "beach": request.beach_name,
            "predicted_count": round(predicted_count, 1),
            "model": {
                "path": request.model_path,
                "type": model_data.get('model_type')
            },
            "features_used": len(available),
            "features_missing": missing if missing else None,
            "weather_provided": request.weather is not None,
            "temporal_features": {
                "hour": int(df['hour'].iloc[0]),
                "day_of_week": int(df['day_of_week'].iloc[0]),
                "month": int(df['month'].iloc[0]),
                "is_weekend": bool(df['is_weekend'].iloc[0])
            }
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/forecast/range")
def forecast_crowd_range(request: MultiForecastRequest):
    """
    Predict crowd counts for multiple hours.

    Example: Forecast tomorrow 8am to 8pm
    {
        "model_path": "models/prediction/crowd_xgboost.pkl",
        "lat": 39.534686,
        "lon": 2.717979,
        "start_datetime": "2024-01-29T08:00:00",
        "hours": 12
    }
    """
    if not HAS_TRAINING:
        raise HTTPException(500, "Training module not available")

    import pandas as pd
    from datetime import timedelta

    try:
        model_data = load_trained_model(request.model_path)
        features = model_data['features']

        # Identify feature types
        weather_features = [f for f in features if f.startswith('ae_') or f.startswith('om_')]
        temporal_features = [f for f in features if f in ['hour', 'day_of_week', 'month', 'is_weekend',
                                                          'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                                                          'dow_sin', 'dow_cos', 'day_of_year']]

        start_dt = datetime.fromisoformat(request.start_datetime.replace(" ", "T"))

        predictions = []

        for i in range(request.hours):
            target_dt = start_dt + timedelta(hours=i)

            row = {
                'datetime': target_dt,
                'lat': request.lat,
                'lon': request.lon
            }

            # Get weather for this hour
            weather_used = {}
            if request.weather_forecasts and i < len(request.weather_forecasts):
                weather_used = request.weather_forecasts[i].copy()
                row.update(request.weather_forecasts[i])
            else:
                try:
                    weather = get_weather_at_point(request.lat, request.lon, target_dt, use_cache=True)
                    if weather:
                        weather_used = weather.copy()
                        row.update(weather)
                except:
                    pass

            df = pd.DataFrame([row])
            df = add_temporal_features(df)

            # Track which features were provided vs filled with 0
            features_provided = [f for f in features if f in df.columns and df[f].iloc[0] != 0]
            features_missing = []

            # Fill missing features
            for f in features:
                if f not in df.columns:
                    df[f] = 0.0
                    features_missing.append(f)

            pred = predict_with_model(model_data, df)
            count = max(0, float(pred[0]))

            prediction_entry = {
                "datetime": target_dt.isoformat(),
                "hour": target_dt.hour,
                "day_of_week": target_dt.weekday(),
                "is_weekend": target_dt.weekday() >= 5,
                "predicted_count": round(count, 1),
                "weather_used": {k: v for k, v in weather_used.items() if k in weather_features} if weather_used else None
            }

            predictions.append(prediction_entry)

        # Summary stats
        counts = [p['predicted_count'] for p in predictions]
        peak_idx = counts.index(max(counts))

        return {
            "beach": request.beach_name,
            "lat": request.lat,
            "lon": request.lon,
            "model": {
                "path": request.model_path,
                "type": model_data.get('model_type'),
                "total_features": len(features),
                "weather_features": weather_features,
                "temporal_features": temporal_features
            },
            "forecast_start": start_dt.isoformat(),
            "forecast_hours": request.hours,
            "summary": {
                "min_count": round(min(counts), 1),
                "max_count": round(max(counts), 1),
                "avg_count": round(sum(counts) / len(counts), 1),
                "peak_hour": predictions[peak_idx]['datetime'],
                "peak_count": predictions[peak_idx]['predicted_count']
            },
            "hourly_predictions": predictions
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/forecast/features")
def get_forecast_features(model_path: str = Query(...)):
    """Get the features required by a trained model for forecasting."""
    if not HAS_TRAINING:
        raise HTTPException(500, "Training module not available")

    try:
        model_data = load_trained_model(model_path)
        features = model_data['features']

        weather_features = [f for f in features if f.startswith('ae_') or f.startswith('om_')]
        temporal_features = [f for f in features if f in ['hour', 'day_of_week', 'month', 'is_weekend',
                                                          'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                                                          'dow_sin', 'dow_cos', 'day_of_year']]
        location_features = [f for f in features if f in ['lat', 'lon']]

        return {
            "model_path": model_path,
            "model_type": model_data.get('model_type'),
            "total_features": len(features),
            "features": {
                "weather": weather_features,
                "temporal": temporal_features,
                "location": location_features
            },
            "feature_importance": model_data.get('feature_importance'),
            "note": "Temporal features are auto-generated from datetime. Provide weather features manually or let API fetch forecast."
        }
    except Exception as e:
        raise HTTPException(400, str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level='debug', timeout_keep_alive=3000)