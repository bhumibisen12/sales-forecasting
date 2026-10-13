"""
app.py
------
Unified API and Execution Entrypoint for AI CRM Sales Forecasting.
Can be run as:
  1. CLI Tool : python app.py
  2. Web API  : uvicorn app:app --port 8000
  3. Submodule: from app import app as sales_forecast_router
"""

from __future__ import annotations
import sys
import shutil
import atexit
from pathlib import Path

# Disable Python bytecode caching
sys.dont_write_bytecode = True

_BASE_DIR = Path(__file__).resolve().parent

def _cleanup_pycache(target_dir=_BASE_DIR, _rmtree=shutil.rmtree):
    try:
        if target_dir is not None and target_dir.exists():
            for p in target_dir.rglob("__pycache__"):
                if p.is_dir():
                    _rmtree(p, ignore_errors=True)
    except BaseException:
        pass

_cleanup_pycache()
atexit.register(_cleanup_pycache)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import Unified Engine
from sales_forecaster import (
    SalesForecaster,
    forecast_next_month,
    forecast_future,
    get_model_metrics,
    get_engine,
)

# ---------------------------------------------------------------------------
# FASTAPI APP (EASY TO MOUNT IN CENTRAL CRM GATEWAY)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI CRM Sales Forecasting Service",
    description="Unified API providing multi-month sales forecasts and pipeline health metrics.",
    version="2.0.0",
)


class ForecastRequest(BaseModel):
    horizon_months: int = 3
    model_source: str | None = None


@app.get("/health")
def health_check():
    """Health check endpoint for CRM service orchestrator."""
    engine = get_engine()
    return {
        "status": "healthy",
        "service": "sales_forecasting_ai",
        "model_loaded": engine.model is not None,
    }


@app.get("/model-metrics")
def model_metrics():
    """Returns trained model accuracy benchmarks and holdout test scores."""
    return {
        "status": "success",
        "metrics": get_model_metrics(),
    }


@app.post("/sales-forecast")
def get_forecast(req: ForecastRequest):
    """Generates multi-month sales forecast with core 5 metrics."""
    try:
        engine = SalesForecaster(model_source=req.model_source) if req.model_source else get_engine()
        results = engine.forecast_future(horizon=req.horizon_months)
        next_month = results[0] if results else {}

        return {
            "status": "success",
            "horizon_months": req.horizon_months,
            "core_metrics": {
                "Predicted Monthly Revenue": next_month.get("Predicted Monthly Revenue"),
                "Pipeline Health Score": next_month.get("Pipeline Health Score"),
                "Deal Closure Trend": next_month.get("Deal Closure Trend"),
                "Revenue Growth Trend": next_month.get("Revenue Growth Trend"),
                "Forecast Confidence Score": next_month.get("Forecast Confidence Score"),
            },
            "forecast": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sales-forecast/next-month")
def get_next_month():
    """Generates next month forecast snapshot."""
    try:
        result = forecast_next_month()
        return {
            "status": "success",
            "core_metrics": {
                "Predicted Monthly Revenue": result["Predicted Monthly Revenue"],
                "Pipeline Health Score": result["Pipeline Health Score"],
                "Deal Closure Trend": result["Deal Closure Trend"],
                "Revenue Growth Trend": result["Revenue Growth Trend"],
                "Forecast Confidence Score": result["Forecast Confidence Score"],
            },
            "forecast": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# CLI RUNNER (FOR STANDALONE TERMINAL EXECUTION)
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 80)
    print("                     AI CRM - SALES FORECASTING SYSTEM")
    print("                   UNIFIED ENTERPRISE PREDICTION ENGINE")
    print("=" * 80)

    engine = get_engine()

    # 1. MODEL ACCURACY & PERFORMANCE SECTION
    metrics = engine.get_model_metrics()
    print("\n" + "-" * 80)
    print("  MODEL ACCURACY & VALIDATION BENCHMARKS")
    print("-" * 80)
    print(f"  Selected Algorithm         : {metrics['model_name']}")
    print(f"  Evaluation Benchmark       : {metrics['benchmark_period']}")
    print(f"  1. R2 Score (Variance)     : {metrics['r2_score']:.4f} ({metrics['r2_percent']} Sales Pattern Captured)")
    print(f"  2. Model Overall Accuracy  : {metrics['accuracy']} (MAPE: {metrics['mape']}%)")
    print(f"  3. Mean Absolute Error     : {metrics['mae']}")
    print(f"  4. Root Mean Squared Error : {metrics['rmse']}")
    print(f"  5. Residual Uncertainty    : {metrics['residual_uncertainty']}")
    print(f"  Total Features Used        : {metrics['feature_count']} Engineered Features")
    print("-" * 80)

    # 2. CORE FORECASTING METRICS (NEXT MONTH)
    res = engine.forecast_next_month()
    print("\n" + "-" * 80)
    print(f"  CORE SALES FORECASTING METRICS (Target: {res['forecast_month']})")
    print("-" * 80)
    print(f"  1. Predicted Monthly Revenue  : {res['Predicted Monthly Revenue']}")
    print(f"  2. Pipeline Health Score      : {res['Pipeline Health Score']}")
    print(f"  3. Deal Closure Trend         : {res['Deal Closure Trend']}")
    print(f"  4. Revenue Growth Trend       : {res['Revenue Growth Trend']}")
    print(f"  5. Forecast Confidence Score  : {res['Forecast Confidence Score']}")
    print("-" * 80)

    # 3. MULTI-MONTH FORECAST TABLE
    print("\nMULTI-MONTH OPERATIONAL FORECAST (3-MONTH HORIZON):")
    multi_res = engine.forecast_future(horizon=3)
    for f in multi_res:
        print(f"  Month: {f['forecast_month']:<15} | Revenue: {f['Predicted Monthly Revenue']:<20} "
              f"| Range: [Rs. {f['revenue_lower_bound']:>11,.2f} - Rs. {f['revenue_upper_bound']:>11,.2f}] "
              f"| Conf: {f['confidence_score']:.1f}%")
    print("-" * 80)

    # 4. MULTIVARIATE OPPORTUNITY DRIVERS
    factors = res.get("top_factors", [])
    if factors:
        feature_labels = {
            "expected_deal_revenue": "Weighted Pipeline Capitalization",
            "pipeline_quality_value": "Lead Qualification & Propensity Index",
            "Deal Value": "Average Contract Value (ACV) Expansion",
            "log_deal_value": "Portfolio Deal Size Distribution",
            "pipeline_conversion_strength": "Pipeline Conversion Kinetics",
        }
        print("\nTOP OPPORTUNITY GROWTH DRIVERS & ATTRIBUTION:")
        for idx, fact in enumerate(factors, 1):
            name = feature_labels.get(fact["feature"], fact["feature"].replace("_", " ").title())
            imp = fact.get("importance", 0)
            lift = fact.get("impact_on_revenue", 0)
            print(f"  {idx}. {name:<40} : {imp:>5.1f}% (Attributed lift: +Rs. {lift:,})")
        print("-" * 80)

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()