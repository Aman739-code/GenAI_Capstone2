import sys
import logging
from config.settings import get_api_key
from models.forecast import generate_forecast
from agent.graph import run_pipeline

logging.basicConfig(level=logging.ERROR)

print("Starting pipeline test...")
try:
    print("Generating forecast...")
    forecast_data = generate_forecast(days=7)
    
    print("Running pipeline...")
    result = run_pipeline(forecast_data)
    
    print("Errors in pipeline result:")
    if result.get("error_log"):
        for err in result["error_log"]:
            print(f"- {err}")
    else:
        print("No errors in error_log.")
except Exception as e:
    import traceback
    traceback.print_exc()
