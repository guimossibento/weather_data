from time import sleep

from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel
import uvicorn
import io
import csv

from weather_module import get_weather_at_point, get_station_info, reload_aemet_data, check_openmeteo_connectivity
from crowd_module import (
    predict_count, load_model, get_model_info, parse_datetime_from_filename,
    load_prediction_from_cache, save_prediction_to_cache, clear_prediction_cache, list_cache_names,
    register_predictor, set_active_predictor, get_predictor, BasePredictor
)

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
        target_dt = datetime.fromisoformat(datetime_str.replace(" ", "T"))
    else:
        target_dt = parse_datetime_from_filename(filename)
        if target_dt is None:
            raise HTTPException(400, "Could not parse datetime from filename")

    # Use specified model or active model
    if model_name:
        try:
            predictor = get_predictor(model_name)
            count = predictor.predict(image_data)
        except ValueError as e:
            raise HTTPException(404, str(e))
    else:
        count = predict_count(image_data)

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
        model_name: Optional[str] = Query(None, alias="model", description="Model to use for prediction"),
        use_cache: bool = Query(True),
        output_format: str = Query("json")
):
    results = []

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
                    cached["index"] = idx
                    results.append(cached)
                    continue

            image_data = await file.read()
            target_dt = parse_datetime_from_filename(filename)

            if target_dt is None:
                results.append({
                    "index": idx,
                    "filename": filename,
                    "name": name,
                    "error": "Could not parse datetime"
                })
                continue

            count = predictor.predict(image_data) if predictor else predict_count(image_data)
            weather_data = get_weather_at_point(lat, lon, target_dt, use_cache=True)

            row = {
                "index": idx,
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
                "index": idx,
                "filename": file.filename,
                "name": name,
                "error": str(e)
            })

    success_count = len([r for r in results if "error" not in r])
    error_count = len([r for r in results if "error" in r])

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
                    cached["index"] = idx
                    results.append(cached)
                    continue

            if not path.exists():
                results.append({
                    "index": idx,
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
                        "index": idx,
                        "path": img.path,
                        "filename": filename,
                        "name": img.name,
                        "error": "Could not parse datetime"
                    })
                    continue

            count = predictor.predict(path) if predictor else predict_count(path)
            weather_data = get_weather_at_point(img.lat, img.lon, target_dt, use_cache=True)

            row = {
                "index": idx,
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
                "index": idx,
                "path": img.path,
                "filename": Path(img.path).name if img.path else None,
                "name": img.name,
                "error": str(e)
            })

    success_count = len([r for r in results if "error" not in r])
    error_count = len([r for r in results if "error" in r])

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
        output_format: str = Query("json")
):
    """Process all images in a directory, replicating subfolder structure in cache."""
    import glob

    dir_path = Path(directory)
    if not dir_path.exists():
        raise HTTPException(404, "Directory not found")

    # Find images (recursive or not)
    if recursive:
        image_files = sorted(glob.glob(str(dir_path / "**" / pattern), recursive=True))
    else:
        image_files = sorted(glob.glob(str(dir_path / pattern)))

    if not image_files:
        raise HTTPException(404, f"No files matching {pattern}")

    # Get predictor once for batch
    if model_name:
        try:
            predictor = get_predictor(model_name)
        except ValueError as e:
            raise HTTPException(404, str(e))
    else:
        predictor = None

    used_model = model_name or get_model_info()["active"]

    results = []
    for idx, img_path in enumerate(image_files):
        try:
            path = Path(img_path)
            filename = path.name

            # Get relative subpath from source directory
            rel_path = path.parent.relative_to(dir_path)
            subpath = str(rel_path) if str(rel_path) != "." else None

            # Check cache first
            if use_cache:
                cached = load_prediction_from_cache(filename, lat, lon, model=used_model, name=cache_name, subpath=subpath)
                if cached:
                    cached["from_cache"] = True
                    cached["index"] = idx
                    results.append(cached)
                    continue

            if not path.exists():
                results.append({
                    "index": idx,
                    "path": img_path,
                    "filename": filename,
                    "subpath": subpath,
                    "cache_name": cache_name,
                    "error": "File not found"
                })
                continue

            target_dt = parse_datetime_from_filename(filename)

            if target_dt is None:
                results.append({
                    "index": idx,
                    "path": img_path,
                    "filename": filename,
                    "subpath": subpath,
                    "cache_name": cache_name,
                    "error": "Could not parse datetime"
                })
                continue

            count = predictor.predict(path) if predictor else predict_count(path)
            weather_data = get_weather_at_point(lat, lon, target_dt, use_cache=True)

            row = {
                "index": idx,
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
            results.append(row)

            # Save to cache with subfolder structure
            if use_cache:
                save_prediction_to_cache(filename, lat, lon, row, model=used_model, name=cache_name, subpath=subpath)
        except Exception as e:
            results.append({
                "index": idx,
                "path": img_path,
                "filename": Path(img_path).name if img_path else None,
                "cache_name": cache_name,
                "error": str(e)
            })

    success_count = len([r for r in results if "error" not in r])
    error_count = len([r for r in results if "error" in r])

    if output_format == "csv":
        return _to_csv_response(results, f"{cache_name}_predictions.csv")

    return {"directory": directory, "cache_name": cache_name, "count": len(results), "success": success_count, "errors": error_count, "results": results}


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
def get_models():
    """Get all registered models and active model."""
    return get_model_info()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level='debug')