import sys

import requests
import json
import glob
import hashlib
from pathlib import Path
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import LinearNDInterpolator
from sklearn.neighbors import KNeighborsRegressor

PROJECT_ROOT = Path.cwd()
AEMET_DIR = PROJECT_ROOT / 'aemet/2022'
CACHE_DIR = PROJECT_ROOT / 'cache/weather'
RESULT_CACHE_DIR = PROJECT_ROOT / 'cache/weather_results'
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


def _load_aemet_data():
    json_files = sorted(glob.glob(str(AEMET_DIR / '*.json')))

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
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df['fint'] = pd.to_datetime(df['fint'])

    # Filter by bounding box
    df_bal = df[
        (df['lat'] >= BBOX['lat_min']) & (df['lat'] <= BBOX['lat_max']) &
        (df['lon'] >= BBOX['lon_min']) & (df['lon'] <= BBOX['lon_max'])
        ].copy()

    # Filter by altitude if column exists
    if 'alt' in df_bal.columns:
        df_bal = df_bal[df_bal['alt'] <= ALT_THRESHOLD].copy()

    return df_bal


df_bal = _load_aemet_data()


def _parse_datetime(target_dt):
    """Parse input to datetime object."""
    if isinstance(target_dt, str):
        return pd.to_datetime(target_dt)
    elif isinstance(target_dt, date) and not isinstance(target_dt, datetime):
        return datetime.combine(target_dt, datetime.min.time())
    return target_dt


def _get_result_cache_key(lat, lon, target_dt):
    """Cache key includes hour for hourly precision."""
    dt_str = target_dt.strftime('%Y-%m-%d_%H')
    return hashlib.md5(f"{lat:.4f}_{lon:.4f}_{dt_str}".encode()).hexdigest()


def _load_from_cache(lat, lon, target_dt):
    cache_key = _get_result_cache_key(lat, lon, target_dt)
    cache_file = RESULT_CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None


def _save_to_cache(lat, lon, target_dt, result):
    cache_key = _get_result_cache_key(lat, lon, target_dt)
    cache_file = RESULT_CACHE_DIR / f"{cache_key}.json"

    with open(cache_file, 'w') as f:
        json.dump(result, f)


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


def _get_aemet_var(df, lat, lon, target_dt, var, time_window_hours=1):
    """Get AEMET variable with time window filtering."""
    if var not in df.columns:
        return None

    target_dt = _parse_datetime(target_dt)

    # Filter by time window around target datetime
    time_start = target_dt - timedelta(hours=time_window_hours)
    time_end = target_dt + timedelta(hours=time_window_hours)

    time_data = df[(df['fint'] >= time_start) & (df['fint'] <= time_end)]

    # Fallback to day if no data in time window
    if len(time_data) == 0:
        time_data = df[df['fint'].dt.date == target_dt.date()]

    if len(time_data) == 0:
        return None

    stations = time_data.groupby(['idema', 'lat', 'lon']).agg({var: 'mean'}).reset_index()

    missing = stations[var].isna()
    if missing.any() and (~missing).sum() >= 2:
        knn = KNeighborsRegressor(n_neighbors=min(3, (~missing).sum()), weights='distance')
        knn.fit(stations.loc[~missing, ['lon', 'lat']].values, stations.loc[~missing, var].values)
        stations.loc[missing, var] = knn.predict(stations.loc[missing, ['lon', 'lat']].values)

    stations = stations.dropna(subset=[var])
    if len(stations) == 0:
        return None

    if len(stations) < 3:
        return float(stations[var].mean())

    points = stations[['lon', 'lat']].values
    values = stations[var].values
    target = np.array([[lon, lat]])

    return _hull_multipoint_interpolate(points, values, target)


def _get_openmeteo(lat, lon, target_dt):
    """Get OpenMeteo data for specific hour."""
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
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except:
            return {}

    if 'hourly' not in data:
        return {}

    times = pd.to_datetime(data['hourly']['time'])

    # Filter for specific hour (+/- 1 hour window)
    mask = (times.date == target_date) & (abs(times.hour - target_hour) <= 1)

    # Fallback to full day if no data in window
    if not mask.any():
        mask = times.date == target_date

    result = {}
    for var in OPENMETEO_VARS:
        if var in data['hourly']:
            vals = np.array(data['hourly'][var])[mask]
            vals = vals[~pd.isna(vals)]
            if len(vals) > 0:
                result[var] = float(np.mean(vals))
    return result


def get_weather_at_point(lat, lon, target_dt, use_cache=True, time_window_hours=1):
    """
    Get all weather variables at any coordinate for a specific datetime.

    Parameters:
        lat, lon: target coordinates
        target_dt: datetime object, date object, or string (e.g. '2022-10-26 14:30')
        use_cache: if True, check/save result cache
        time_window_hours: hours before/after target time to include for AEMET

    Returns:
        dict with 'ae_*' (AEMET) and 'om_*' (OpenMeteo) keys
    """
    target_dt = _parse_datetime(target_dt)

    if use_cache:
        cached = _load_from_cache(lat, lon, target_dt)
        if cached is not None:
            print(f"Using cached weather for ({lat}, {lon}) at {target_dt}")
            return cached

    result = {}

    for var in AEMET_VARS:
        val = _get_aemet_var(df_bal, lat, lon, target_dt, var, time_window_hours)
        if val is not None:
            result[f'ae_{var}'] = val

    om = _get_openmeteo(lat, lon, target_dt)
    for var, val in om.items():
        result[f'om_{var}'] = val

    if use_cache and result:
        _save_to_cache(lat, lon, target_dt, result)

    return result


def get_station_info():
    """Get info about loaded stations."""
    if len(df_bal) == 0:
        return {'records': 0, 'stations': 0}

    stations = df_bal.groupby('idema').agg({
        'lat': 'first', 'lon': 'first'
    }).reset_index()

    if 'alt' in df_bal.columns:
        alt_info = df_bal.groupby('idema')['alt'].first()
        stations = stations.merge(alt_info, on='idema')

    return {
        'records': len(df_bal),
        'stations': df_bal['idema'].nunique(),
        'date_range': (df_bal['fint'].min(), df_bal['fint'].max()),
        'alt_threshold': ALT_THRESHOLD
    }


if __name__ == '__main__':
    # info = get_station_info()
    # print(f"AEMET records: {info['records']:,}")
    # print(f"Stations: {info['stations']}")
    # print(f"Altitude threshold: {info['alt_threshold']}m")

    # Get coordinates from arguments or use default
    if len(sys.argv) >= 3:
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
    else:
        lat, lon = 39.7325522, 3.2400431  # Default coordinates

    # Test with date string
    target_date = datetime(2022, 10, 26, 22, 30)
    # Test with datetime
    weather = get_weather_at_point(
        lat=lat,
        lon=lon,
        target_dt=target_date
    )
    print(f"Weather at ({lat}, {lon}) on {target_date}:")
    print(json.dumps(weather, indent=2))