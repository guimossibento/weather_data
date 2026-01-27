import sys
import requests
import json
import glob
import hashlib
import pickle
from pathlib import Path
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import LinearNDInterpolator
from sklearn.neighbors import KNeighborsRegressor

PROJECT_ROOT = Path.cwd()
AEMET_DIR = PROJECT_ROOT / 'aemet'
CACHE_DIR = PROJECT_ROOT / 'cache/weather'
RESULT_CACHE_DIR = PROJECT_ROOT / 'cache/weather_results'
AEMET_CACHE_FILE = CACHE_DIR / 'aemet_data.pkl'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

BBOX = {'lat_min': 38.5, 'lat_max': 40.5, 'lon_min': 1.0, 'lon_max': 4.5}
ALT_THRESHOLD = 300

AEMET_VARS = ['ta', 'hr', 'prec', 'vv', 'dv', 'pres', 'tamin', 'tamax', 'inso', 'pres_nmar', 'ts']

OPENMETEO_VARS = [
    'temperature_2m', 'apparent_temperature', 'dewpoint_2m', 'relative_humidity_2m',
    'pressure_msl', 'precipitation', 'rain', 'wind_speed_10m', 'wind_direction_10m',
    'wind_gusts_10m', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
    'visibility', 'uv_index', 'uv_index_clear_sky', 'sunshine_duration',
    'evapotranspiration', 'vapour_pressure_deficit'
]

# Lazy loaded data
_df_bal = None


def _load_aemet_data(force_reload=False):
    """Load AEMET data with pickle cache."""
    global _df_bal

    if _df_bal is not None and not force_reload:
        return _df_bal

    # Try pickle cache first
    if AEMET_CACHE_FILE.exists() and not force_reload:
        try:
            _df_bal = pd.read_pickle(AEMET_CACHE_FILE)
            return _df_bal
        except:
            pass

    # Load from JSON files (all years)
    json_files = sorted(glob.glob(str(AEMET_DIR / '**/*.json'), recursive=True))

    all_records = []
    for f in json_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.loads(file.read())
                if 'datos' in data:
                    all_records.extend(data['datos'])
        except:
            pass

    if not all_records:
        _df_bal = pd.DataFrame()
        return _df_bal

    df = pd.DataFrame(all_records)
    df['fint'] = pd.to_datetime(df['fint'])

    _df_bal = df[
        (df['lat'] >= BBOX['lat_min']) & (df['lat'] <= BBOX['lat_max']) &
        (df['lon'] >= BBOX['lon_min']) & (df['lon'] <= BBOX['lon_max'])
        ].copy()

    if 'alt' in _df_bal.columns:
        _df_bal = _df_bal[_df_bal['alt'] <= ALT_THRESHOLD].copy()

    # Save pickle cache
    try:
        _df_bal.to_pickle(AEMET_CACHE_FILE)
    except:
        pass

    return _df_bal


def _parse_datetime(target_dt):
    if isinstance(target_dt, str):
        return pd.to_datetime(target_dt)
    elif isinstance(target_dt, date) and not isinstance(target_dt, datetime):
        return datetime.combine(target_dt, datetime.min.time())
    return target_dt


def _get_cache_key(lat, lon, target_dt):
    dt_str = target_dt.strftime('%Y-%m-%d_%H')
    return hashlib.md5(f"{lat:.4f}_{lon:.4f}_{dt_str}".encode()).hexdigest()


def _load_from_cache(lat, lon, target_dt):
    cache_file = RESULT_CACHE_DIR / f"{_get_cache_key(lat, lon, target_dt)}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return None


def _save_to_cache(lat, lon, target_dt, result):
    cache_file = RESULT_CACHE_DIR / f"{_get_cache_key(lat, lon, target_dt)}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except:
        pass


def _hull_multipoint_interpolate(points, values, target):
    linear = LinearNDInterpolator(points, values)
    delaunay = Delaunay(points)

    if delaunay.find_simplex(target)[0] >= 0:
        return float(linear(target)[0])

    hull = ConvexHull(points)
    hull_closed = np.vstack([points[hull.vertices], points[hull.vertices[0]]])
    edges = np.diff(hull_closed, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    total = lengths.sum()

    boundary_pts, cum, idx = [], 0, 0
    for d in np.linspace(0, total, 50, endpoint=False):
        while idx < len(lengths) - 1 and cum + lengths[idx] < d:
            cum += lengths[idx]
            idx += 1
        t = (d - cum) / lengths[idx] if lengths[idx] > 0 else 0
        boundary_pts.append(hull_closed[idx] + np.clip(t, 0, 1) * edges[idx])

    boundary_pts = np.array(boundary_pts)
    boundary_vals = linear(boundary_pts)
    valid = ~np.isnan(boundary_vals)
    boundary_pts, boundary_vals = boundary_pts[valid], boundary_vals[valid]

    if len(boundary_pts) < 2:
        return None

    knn = KNeighborsRegressor(n_neighbors=min(10, len(boundary_pts)), weights='distance')
    knn.fit(boundary_pts, boundary_vals)
    return float(knn.predict(target)[0])


def _get_all_aemet_vars(lat, lon, target_dt, time_window_hours=1):
    """Get ALL AEMET variables in one pass (faster than per-variable)."""
    df = _load_aemet_data()
    if len(df) == 0:
        print("[AEMET] No data loaded")
        return {}

    target_dt = _parse_datetime(target_dt)
    time_start = target_dt - timedelta(hours=time_window_hours)
    time_end = target_dt + timedelta(hours=time_window_hours)

    time_data = df[(df['fint'] >= time_start) & (df['fint'] <= time_end)]
    if len(time_data) == 0:
        time_data = df[df['fint'].dt.date == target_dt.date()]
    if len(time_data) == 0:
        print(f"[AEMET] No data for {target_dt.date()}")
        return {}

    available_vars = [v for v in AEMET_VARS if v in time_data.columns]
    if not available_vars:
        return {}

    agg_dict = {v: 'mean' for v in available_vars}
    stations = time_data.groupby(['idema', 'lat', 'lon']).agg(agg_dict).reset_index()

    result = {}
    target = np.array([[lon, lat]])

    for var in available_vars:
        var_stations = stations[['lon', 'lat', var]].dropna()
        if len(var_stations) == 0:
            continue

        if len(var_stations) < 3:
            result[f'ae_{var}'] = round(float(var_stations[var].mean()), 4)
            continue

        points = var_stations[['lon', 'lat']].values
        values = var_stations[var].values

        try:
            val = _hull_multipoint_interpolate(points, values, target)
            if val is not None:
                result[f'ae_{var}'] = round(val, 4)
        except Exception as e:
            result[f'ae_{var}'] = round(float(values.mean()), 4)

    return result


def _get_openmeteo(lat, lon, target_dt):
    target_dt = _parse_datetime(target_dt)
    target_date = target_dt.date()
    target_hour = target_dt.hour

    start = target_date - timedelta(days=1)
    end = target_date + timedelta(days=1)

    cache_key = hashlib.md5(f"{lat:.4f}_{lon:.4f}_{start}_{end}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"om_{cache_key}.json"

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            data = json.load(f)
    else:
        try:
            resp = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    'latitude': lat, 'longitude': lon,
                    'start_date': str(start), 'end_date': str(end),
                    'hourly': ','.join(OPENMETEO_VARS),
                    'timezone': 'Europe/Madrid'
                },
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            # Check for API error response
            if 'error' in data:
                print(f"[OpenMeteo] API error: {data.get('reason', 'Unknown')}")
                return {}

            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except requests.exceptions.RequestException as e:
            print(f"[OpenMeteo] Request failed: {e}")
            return {}
        except Exception as e:
            print(f"[OpenMeteo] Error: {e}")
            return {}

    if 'hourly' not in data:
        return {}

    times = pd.to_datetime(data['hourly']['time'])
    mask = (times.date == target_date) & (abs(times.hour - target_hour) <= 1)
    if not mask.any():
        mask = times.date == target_date

    result = {}
    for var in OPENMETEO_VARS:
        if var in data['hourly']:
            vals = np.array(data['hourly'][var])[mask]
            vals = vals[~pd.isna(vals)]
            if len(vals) > 0:
                result[var] = round(float(np.mean(vals)), 4)
    return result


def get_weather_at_point(lat, lon, target_dt, use_cache=True, time_window_hours=1):
    """
    Get all weather variables at any coordinate.

    Parameters:
        lat, lon: target coordinates
        target_dt: datetime, date, or string
        use_cache: use pickle cache for results
        time_window_hours: AEMET time window

    Returns:
        dict with 'ae_*' (AEMET) and 'om_*' (OpenMeteo) keys
    """
    target_dt = _parse_datetime(target_dt)

    if use_cache:
        cached = _load_from_cache(lat, lon, target_dt)
        if cached is not None:
            return cached

    result = _get_all_aemet_vars(lat, lon, target_dt, time_window_hours)

    om = _get_openmeteo(lat, lon, target_dt)
    for var, val in om.items():
        result[f'om_{var}'] = val

    # Always save to cache if we got any result
    if use_cache and result:
        _save_to_cache(lat, lon, target_dt, result)
        print(f"[Weather] Cached result for ({lat:.4f}, {lon:.4f}) at {target_dt}")

    return result


def reload_aemet_data():
    """Force reload AEMET data from JSON files."""
    return _load_aemet_data(force_reload=True)


def check_openmeteo_connectivity():
    """Check if OpenMeteo API is accessible."""
    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={'latitude': 39.5, 'longitude': 2.5, 'start_date': '2022-07-15', 'end_date': '2022-07-16', 'hourly': 'temperature_2m'},
            timeout=10
        )
        return {"status": "ok", "code": resp.status_code}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def get_station_info():
    df = _load_aemet_data()
    if len(df) == 0:
        return {'records': 0, 'stations': 0}

    # Get available years from data
    years = sorted(df['fint'].dt.year.unique().tolist())

    return {
        'records': len(df),
        'stations': df['idema'].nunique(),
        'years': years,
        'date_range': (df['fint'].min(), df['fint'].max()),
        'alt_threshold': ALT_THRESHOLD
    }


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        lat, lon = float(sys.argv[1]), float(sys.argv[2])
    else:
        lat, lon = 39.7325522, 3.2400431

    target_date = datetime(2022, 10, 26, 22, 30)

    import time

    t0 = time.time()
    weather = get_weather_at_point(lat, lon, target_date)
    t1 = time.time()

    print(f"Weather at ({lat}, {lon}) on {target_date}:")
    print(json.dumps(weather, indent=2))
    print(f"\nTime: {t1 - t0:.3f}s")