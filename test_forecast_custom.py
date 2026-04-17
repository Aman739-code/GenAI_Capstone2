import pandas as pd
from models.forecast import save_sample_data, generate_forecast

# 1. Test sample data generation
print("Generatings sample data...")
csv_path = save_sample_data()
print("Sample data saved at:", csv_path)

# 2. Test reading and passing to forecast
print("\nTesting forecast with custom df...")
df = pd.read_csv(csv_path)

# Take 15 rows to test min length (10)
subset_df = df.head(15).copy()

try:
    forecast_data = generate_forecast(days=3, custom_df=subset_df)
    print("Forecast generated successfully!")
    print("Hourly predictions:", len(forecast_data["hourly_predictions"]))
    print("Model metrics:", forecast_data["model_metrics"])
except Exception as e:
    import traceback
    traceback.print_exc()

