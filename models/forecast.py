"""
Milestone 1 Synthetic Solar Forecast Model.
Generates realistic solar irradiance data and trains a RandomForestRegressor
to produce forecast outputs that feed into the agentic pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def _generate_synthetic_solar_data(days: int = 90) -> pd.DataFrame:
    """
    Generate realistic synthetic solar irradiance data.
    Models diurnal cycles, seasonal variation, cloud cover effects,
    and temperature correlations.
    """
    rng = np.random.default_rng(42)
    hours = days * 24
    timestamps = [datetime(2025, 1, 1) + timedelta(hours=h) for h in range(hours)]

    data = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_year = ts.timetuple().tm_yday

        # Solar elevation angle approximation (diurnal cycle)
        solar_angle = max(0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0.0

        # Seasonal modifier (higher in summer)
        seasonal_mod = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        # Cloud cover (0-1, with temporal correlation)
        cloud_base = 0.3 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
        cloud_cover = np.clip(cloud_base + rng.normal(0, 0.2), 0, 1)

        # GHI (Global Horizontal Irradiance) in W/m^2
        clear_sky_ghi = 1000 * solar_angle * seasonal_mod
        ghi = clear_sky_ghi * (1 - 0.75 * cloud_cover) + rng.normal(0, 15)
        ghi = max(0, ghi)

        # DNI (Direct Normal Irradiance)
        dni = ghi * (0.8 - 0.5 * cloud_cover) + rng.normal(0, 10)
        dni = max(0, dni)

        # DHI (Diffuse Horizontal Irradiance)
        dhi = ghi * (0.2 + 0.4 * cloud_cover) + rng.normal(0, 5)
        dhi = max(0, dhi)

        # Temperature (°C)
        temp_base = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temp_diurnal = 5 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else -3
        temperature = temp_base + temp_diurnal + rng.normal(0, 2)

        # Wind speed (m/s)
        wind_speed = max(0, 4 + 2 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 1.5))

        # Power output (kW) - simplified PV model for a 10kW system
        efficiency = 0.18 * (1 - 0.004 * max(0, temperature - 25))
        panel_area = 55  # m² for ~10kW system
        power_output = max(0, ghi * panel_area * efficiency / 1000)

        data.append({
            "timestamp": ts,
            "hour": hour,
            "day_of_year": day_of_year,
            "ghi": round(ghi, 2),
            "dni": round(dni, 2),
            "dhi": round(dhi, 2),
            "temperature": round(temperature, 2),
            "cloud_cover": round(cloud_cover, 4),
            "wind_speed": round(wind_speed, 2),
            "power_output_kw": round(power_output, 3),
        })

    return pd.DataFrame(data)


def _train_model(df: pd.DataFrame) -> tuple[RandomForestRegressor, dict]:
    """Train a RandomForestRegressor on the synthetic data and return metrics."""
    features = ["hour", "day_of_year", "temperature", "cloud_cover", "wind_speed", "ghi"]
    target = "power_output_kw"

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }

    return model, metrics


def generate_forecast(days: int = 7, custom_df: pd.DataFrame = None) -> dict:
    """
    Generate a solar energy forecast for the specified number of days.
    This is the main entry point that produces the Milestone 1 output
    consumed by the agentic pipeline.

    Returns:
        dict with keys:
            - hourly_predictions: list of hourly power output values (kW)
            - daily_summaries: list of daily summary dicts
            - model_metrics: dict of model performance metrics
            - metadata: dict of generation metadata
            - raw_features: dict of feature arrays for analysis
    """
    start_date = datetime(2025, 4, 1)
    
    if custom_df is not None:
        required_cols = ["hour", "day_of_year", "temperature", "cloud_cover", "wind_speed", "ghi", "power_output_kw"]
        missing = [c for c in required_cols if c not in custom_df.columns]
        if missing:
            raise ValueError(f"Uploaded CSV is missing required columns: {missing}")
        if len(custom_df) < 10:
            raise ValueError("Uploaded CSV must have at least 10 rows of historical data.")
            
        train_df = custom_df
        
        if "timestamp" in train_df.columns:
            train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
            start_date = train_df["timestamp"].max() + timedelta(hours=1)
    else:
        # Generate training data
        train_df = _generate_synthetic_solar_data(days=90)

    # Train model
    model, metrics = _train_model(train_df)

    # Generate forecast period data
    rng = np.random.default_rng(123)
    forecast_hours = days * 24
    forecast_data = []

    for h in range(forecast_hours):
        ts = start_date + timedelta(hours=h)
        hour = ts.hour
        day_of_year = ts.timetuple().tm_yday

        cloud_cover = np.clip(0.35 + rng.normal(0, 0.25), 0, 1)
        temperature = 20 + 8 * np.sin(np.pi * (hour - 6) / 12) + rng.normal(0, 2) if 6 <= hour <= 18 else 12 + rng.normal(0, 2)
        wind_speed = max(0, 3.5 + rng.normal(0, 1.5))

        solar_angle = max(0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0.0
        ghi = max(0, 1000 * solar_angle * 0.85 * (1 - 0.75 * cloud_cover) + rng.normal(0, 10))

        features = np.array([[hour, day_of_year, temperature, cloud_cover, wind_speed, ghi]])
        prediction = max(0, float(model.predict(features)[0]))

        forecast_data.append({
            "timestamp": ts.isoformat(),
            "hour": hour,
            "day_of_year": day_of_year,
            "predicted_power_kw": round(prediction, 3),
            "ghi": round(ghi, 2),
            "temperature": round(temperature, 2),
            "cloud_cover": round(cloud_cover, 4),
            "wind_speed": round(wind_speed, 2),
        })

    forecast_df = pd.DataFrame(forecast_data)

    # Create daily summaries
    daily_summaries = []
    for day_offset in range(days):
        day_start = day_offset * 24
        day_end = day_start + 24
        day_data = forecast_df.iloc[day_start:day_end]

        daily_gen = day_data["predicted_power_kw"].sum()
        peak_gen = day_data["predicted_power_kw"].max()
        avg_cloud = day_data["cloud_cover"].mean()
        avg_temp = day_data["temperature"].mean()

        daily_summaries.append({
            "date": (start_date + timedelta(days=day_offset)).strftime("%Y-%m-%d"),
            "total_generation_kwh": round(daily_gen, 2),
            "peak_generation_kw": round(peak_gen, 3),
            "avg_cloud_cover": round(avg_cloud, 4),
            "avg_temperature": round(avg_temp, 2),
            "generation_hours": int((day_data["predicted_power_kw"] > 0.1).sum()),
        })

    # Hourly predictions as flat list
    hourly_predictions = forecast_df["predicted_power_kw"].tolist()

    # Feature arrays for analysis
    raw_features = {
        "timestamps": forecast_df["timestamp"].tolist(),
        "ghi": forecast_df["ghi"].tolist(),
        "temperatures": forecast_df["temperature"].tolist(),
        "cloud_cover": forecast_df["cloud_cover"].tolist(),
        "wind_speed": forecast_df["wind_speed"].tolist(),
    }

    return {
        "hourly_predictions": hourly_predictions,
        "daily_summaries": daily_summaries,
        "model_metrics": metrics,
        "metadata": {
            "location": "Sample Solar Farm, Arizona, USA",
            "system_capacity_kw": 10.0,
            "panel_area_m2": 55.0,
            "model_type": "RandomForestRegressor",
            "forecast_days": days,
            "forecast_start": start_date.isoformat(),
            "forecast_end": (start_date + timedelta(days=days)).isoformat(),
            "generated_at": datetime.now().isoformat(),
        },
        "raw_features": raw_features,
    }


def save_sample_data():
    """Save sample solar irradiance data to CSV for reference."""
    from config.settings import DATA_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = _generate_synthetic_solar_data(days=30)
    df.to_csv(DATA_DIR / "solar_irradiance_sample.csv", index=False)
    return DATA_DIR / "solar_irradiance_sample.csv"


if __name__ == "__main__":
    forecast = generate_forecast(days=7)
    print(f"Generated {len(forecast['hourly_predictions'])} hourly predictions")
    print(f"Model Metrics: {json.dumps(forecast['model_metrics'], indent=2)}")
    print(f"Daily Summaries: {len(forecast['daily_summaries'])} days")
