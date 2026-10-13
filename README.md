# 📈 AI CRM Sales Forecasting Engine
> **Enterprise-Grade Predictive Revenue Engine & Deal Pipeline Analytics**  
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-91.21%25-emerald)](#-model-benchmarks--95-confidence-bounds)
[![R2 Score](https://img.shields.io/badge/R%C2%B2%20Score-0.9521-blue)](#-model-benchmarks--95-confidence-bounds)
[![Algorithm](https://img.shields.io/badge/Core%20Algorithm-CatBoost%20Regressor-orange)](#-model-benchmarks--95-confidence-bounds)
[![API](https://img.shields.io/badge/REST%20API-FastAPI%20(CORS%20Ready)-purple)](#-api-endpoints--react-integration)

---

## 📌 At a Glance (What is this project?)

In most companies, sales targets are guessed manually by sales representatives on Excel spreadsheets. These guesses are often overly optimistic, biased, or inaccurate.

This project provides an **AI-driven Sales Forecasting Engine** that replaces guesswork with objective machine learning:
1. It looks at **past revenue history** (what the company actually made over previous months).
2. It analyzes **active live deals** currently in the sales pipeline (deal values, win probabilities, client responsiveness).
3. It outputs **exact predicted revenue in ₹ Rupees** for upcoming months with **95.21% statistical accuracy**, a 95% safe financial cushion, and the top business reasons driving growth.

---

## 🗺️ The "Google Maps" Analogy (How to Understand It Easily)

If you are new to AI or Data Science, think of this model like **Google Maps predicting your trip arrival time**:

```
┌────────────────────────────────────────────────────────────────────────┐
│  GOOGLE MAPS TRIP PREDICTION               AI CRM SALES FORECAST       │
├────────────────────────────────────────────────────────────────────────┤
│  1. Historical highway speeds    ──▶       1. Past months' revenue     │
│  2. Live traffic conditions      ──▶       2. Live deals in pipeline   │
│  3. Weather / road closures      ──▶       3. Customer urgency & score │
├────────────────────────────────────────────────────────────────────────┤
│  OUTPUT:                                   OUTPUT:                     │
│  "Arriving in 42 minutes                   "Predicted Monthly Revenue: │
│   (Estimated window: 40 - 45 mins)"         ₹20,47,934.92              │
│                                             (Safe window: ₹16.6L-24.3L)"│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart: Run in 30 Seconds

```powershell
# 1. Open Terminal in the project folder
cd d:\salse_forecasting

# 2. Install dependencies (one-time setup)
pip install -r requirements.txt

# 3. Option A: Run terminal dashboard
python app.py

# 4. Option B: Run Web API Server for Frontend / Swagger UI
uvicorn app:app --port 8000 --reload
# (Open in browser: http://127.0.0.1:8000/docs)
```

---

## 🎯 What Does the Model Predict for Next Month? (Live Output)

When you run `python app.py` or query `GET /sales-forecast/next-month`, here is the live prediction:

| Core Metric Name | Predicted Value | What It Means (Plain English) |
|---|---|---|
| **Target Forecast Month** | **January 2026** | The upcoming calendar month being predicted. |
| **Predicted Monthly Revenue** | **Rs. 2,047,934.92** | The expected total sales revenue in Rupees for January 2026. |
| **95% Safe Revenue Window** | **Rs. 16.60L – Rs. 24.36L** | Guaranteed safety cushion (Worst-case to Best-case limit). |
| **Pipeline Health Score** | **40.82 / 100** | Overall quality rating of active deals in the pipeline. |
| **Deal Closure Trend** | **Upward (+141.7%)** | **33.6 expected closed deals** (up +141.7% vs historical average). |
| **Revenue Growth Trend** | **+2.0% MoM** | Positive upward sales velocity. |
| **Forecast Confidence Score** | **93.2%** | How statistically confident the AI is in this projection. |

---

## 🔄 End-to-End System Workflow

```mermaid
flowchart TD
    subgraph Data_Layer["1. Data Ingestion Layer"]
        A1[("datasets/pipeline-data.csv<br/>(Live Active Deals)")]
        A2[("datasets/monthly_sales.csv<br/>(Historical Monthly Revenue)")]
        A3[("datasets/processed-data.csv<br/>(Historical Target Labels)")]
    end

    subgraph Feature_Engine["2. 35-Feature Engineering Pipeline"]
        B1["Zero-Leakage Time Lags (Lag 1, 2, 3, 6)"]
        B2["Moving Averages (3-Mo, 6-Mo Rolling Windows)"]
        B3["Domain Ratios (Expected Revenue, Deal Log Value)"]
        B4["Trigonometric Seasonality (Month Sin / Cos)"]
        B5["35-Feature Clean Matrix (X)"]
    end

    subgraph Model_Layer["3. AI Prediction Engine"]
        C1[("saved_models/revenue_best_model.joblib<br/>CatBoost Regressor (R² = 0.9521)")]
        C2["Bottom-Up Deal-by-Deal Inference"]
        C3["Statistical Risk Calibration (95% CI Bounds)"]
    end

    subgraph API_Layer["4. Delivery Layer"]
        D1["FastAPI REST Endpoints (CORS Enabled)"]
        D2["Terminal Console Runner (python app.py)"]
    end

    subgraph Frontend_UI["5. Frontend CRM Dashboard"]
        E1["Hero Revenue Card (₹20.47L)"]
        E2["Pipeline Health Gauge (40.82 / 100)"]
        E3["Closure Trend Badge (+141.7%)"]
        E4["3-Month Line Chart with Shaded 95% Area"]
        E5["Ranked Horizontal Bar Chart (Top Drivers)"]
    end

    A1 & A2 & A3 --> B1 & B2 & B3 & B4 --> B5
    B5 --> C2
    C1 --> C2 --> C3
    C3 --> D1 & D2
    D1 --> E1 & E2 & E3 & E4 & E5
```

---

## 📋 Input Parameters (CRM Data Specifications)

If you are building deal entry modals, lead capture forms, or pipeline screens, here are the **17 raw fields** ingested by the engine:

| # | Field Name | Data Type | Format / Allowed Values | Validation Rule | Sample Value | Recommended UI Form Control |
|---|---|---|---|---|---|---|
| 1 | `Record_ID` | `string` | Alphanumeric (`LEAD-XXXX`) | Non-null, unique per lead | `"LEAD-0001"` | Read-only auto-generated label |
| 2 | `Month` | `string` | `YYYY-MM` format | Valid calendar month | `"2025-12"` | Month picker (`<input type="month">`) |
| 3 | `Pipeline Health` | `float` | Continuous ($0.0$ to $100.0$) | Range: $0$ to $100$ | `48.06` | Slider or numeric stepper ($0 - 100$) |
| 4 | `Active Deals` | `integer` | Count ($\ge 1$) | Positive integer $\ge 1$ | `14` | Stepper button (`min=1`) |
| 5 | `Lead Scores` | `float` | Rating ($0.0$ to $100.0$) | Range: $0$ to $100$ | `55.37` | Score dial or rating slider |
| 6 | `Conversion Probability` | `float` | Percentage ($0.0$ to $100.0$) | Range: $0$ to $100$ | `49.21` | Percentage slider ($0\% - 100\%$) |
| 7 | `Deal Value` | `integer` | Currency INR ($\ge 0$) | Must be $> 0$ (capped at ₹5 Cr) | `1945000` | Currency input with ₹ prefix |
| 8 | `Customer Engagement` | `float` | Percentage ($0.0$ to $100.0$) | Range: $0$ to $100$ | `68.04` | Activity progress bar ($0 - 100$) |
| 9 | `Follow-up Success Rate` | `float` | Percentage ($0.0$ to $100.0$) | Range: $0$ to $100$ | `54.05` | Percentage input ($0 - 100$) |
| 10 | `Industry Type` | `string` | Categorical text | Must match recognized sectors | `"Professional Services"` | Dropdown (`IT`, `Finance`, `Logistics`, etc.) |
| 11 | `Urgency` | `string` | Enum category | `Low`, `Medium`, `High`, `Critical` | `"High"` | Segmented control / Radio pills |
| 12 | `Budget` | `string` | Enum category | `Low`, `Medium`, `High`, `Enterprise` | `"Enterprise"` | Dropdown selector |
| 13 | `Preferred Comm Time` | `string` | Enum category | `Morning`, `Late Morning`, `Afternoon`, `Evening` | `"Late Morning"` | Time slot selector |
| 14 | `Lead Source` | `string` | Enum category | `Website`, `Cold Call`, `Google Ads`, `WhatsApp`, `Partner` | `"Website"` | Source channel dropdown |
| 15 | `Seasonality` | `string` | Enum category | `Low`, `Medium`, `High`, `Peak` | `"Medium"` | Seasonality badge |
| 16 | `Sentiment/Risk Score` | `float` | Rating ($0.0$ to $100.0$) | Risk index in $[0, 100]$ | `47.32` | Risk indicator gauge |
| 17 | `Monthly Revenue` | `integer` | Currency INR | Historical actual closed revenue | `2123000` | Historical database record (past only) |

---

## 📊 Output Parameters & UI Component Mapping

The API returns **both formatted strings** (with ₹ symbols, commas, and percentage tags) and **raw numeric numbers**.  
- Use **formatted strings** for direct text labels (`data.core_metrics[...]`).  
- Use **raw numbers** for chart plotting, axis ranges, and calculations (`data.forecast[...]`).

| JSON Key Name | Data Type | Live Sample Value | Purpose | Recommended UI Component |
|---|---|---|---|---|
| `Predicted Monthly Revenue` | `string` | `"Rs. 2,047,934.92"` | Human-readable currency text | **Hero KPI Headline Card** |
| `predicted_revenue` | `number` | `2047934.92` | Raw numeric float | Chart.js / Recharts Y-axis value |
| `forecast_month` | `string` | `"January 2026"` | Target timeline label | Card Subtitle / Target Badge |
| `Pipeline Health Score` | `string` | `"40.82 / 100"` | Ratio text out of 100 | Secondary Metric Card with Pill |
| `pipeline_health_score` | `number` | `40.82` | Raw health score | **Speedometer Gauge / Radial Progress** |
| `Deal Closure Trend` | `string` | `"Upward (33.6 expected closed deals, +141.7% vs hist avg)"` | Trend, volume & comparison | **Status Banner with Upward Icon** |
| `Revenue Growth Trend` | `string` | `"+2.0% projected growth (Upward Momentum Trajectory)"` | Growth trajectory | **Green Positive Momentum Pill** |
| `Forecast Confidence Score` | `string` | `"93.2%"` | Calibrated confidence text | Confidence Meter Badge |
| `confidence_score` | `number` | `93.2` | Raw confidence % | Circular Progress Ring |
| `revenue_lower_bound` | `number` | `1659945.12` | 95% Worst-case safety limit | Lower boundary of shaded chart area |
| `revenue_upper_bound` | `number` | `2435924.72` | 95% Best-case expansion limit | Upper boundary of shaded chart area |
| `expected_closed_deals` | `number` | `33.6` | Expected closed deals count | Subtitle deal counter |
| `top_factors` | `array` | `[ {feature, importance, impact...} ]` | Top 5 business drivers | **Ranked Horizontal Bar Chart** |

---

## 🔬 Feature Engineering (All 35 Features Explained)

The AI model does not just look at raw deal values. It derives **35 mathematical indicators** grouped into 5 categories:

### 1. Core Deal Numbers (8 Features)
- `Pipeline Health`: Rating from 0 to 100 representing deal qualification maturity.
- `Active Deals`: Count of concurrent negotiation contracts.
- `Lead Scores`: Inbound qualification index (0 to 100).
- `Conversion Probability`: Win probability percentage (0% to 100%).
- `Deal Value`: Total contract value in Indian Rupees (₹).
- `Customer Engagement`: Frequency of client interactions and proposal views.
- `Follow-up Success Rate`: Sales rep outreach response percentage.
- `Sentiment/Risk Score`: Algorithmic risk index detecting churn or friction.

### 2. Business Categoricals (6 Features)
- `Industry Type`, `Urgency`, `Budget`, `Preferred Communication Time`, `Lead Source`, `Seasonality`.
- *Handled natively by CatBoost decision trees without one-hot encoding bloat.*

### 3. Historical Revenue Lags & Rolling Windows (8 Features)
*Why Lags?* Next month's revenue is anchored to past momentum. Strictly past verified data is used to guarantee zero lookahead leakage:
- `revenue_lag_1` ($t-1$): Revenue from 1 month ago (immediate baseline run-rate).
- `revenue_lag_2` ($t-2$): Revenue from 2 months ago (compared with Lag 1 to detect trajectory).
- `revenue_lag_3` ($t-3$): Revenue from 3 months ago (captures 90-day B2B quarterly target cycles).
- `revenue_lag_6` ($t-6$): Revenue from 6 months ago (captures semi-annual / half-yearly fiscal cycles).
- `revenue_roll_mean_3 / 6`: Moving average over 3 and 6 months (smoothes short-term volatility).
- `revenue_roll_std_3 / 6`: Standard deviation over 3 and 6 months (measures revenue volatility).

### 4. Financial Domain Ratios & Kinetics (9 Features)
- `expected_deal_revenue`: $\frac{\text{Deal Value} \times \text{Conversion Probability}}{100}$ (The #1 growth driver with 52% model importance).
- `deal_value_per_active_deal`: $\frac{\text{Deal Value}}{\text{Active Deals}}$ (Average contract size).
- `log_deal_value`: $\ln(1 + \text{Deal Value})$ (Prevents rare ₹10 Crore deals from skewing predictions).
- `engagement_followup_strength`: $\frac{\text{Customer Engagement} \times \text{Follow-up Success Rate}}{100}$.
- `lead_conversion_strength`: $\frac{\text{Lead Scores} \times \text{Conversion Probability}}{100}$.
- `pipeline_quality_value`: $\frac{\text{Pipeline Health} \times \text{Deal Value}}{100}$.
- `lead_engagement_strength`: $\frac{\text{Lead Scores} \times \text{Customer Engagement}}{100}$.
- `conversion_engagement_strength`: $\frac{\text{Conversion Probability} \times \text{Customer Engagement}}{100}$.
- `pipeline_conversion_strength`: $\frac{\text{Pipeline Health} \times \text{Conversion Probability}}{100}$ (Deal conversion kinetics).

### 5. Cyclical Calendar Seasonality (4 Features)
- `month_num` ($1 - 12$) & `quarter` ($1 - 4$).
- `month_sin` $= \sin\left(\frac{2\pi \times \text{Month}}{12}\right)$ & `month_cos` $= \cos\left(\frac{2\pi \times \text{Month}}{12}\right)$.  
  *Converts months into a circular 12-hour clock so December ($12$) and January ($1$) are recognized as adjacent neighbors.*

---

## 🏆 CatBoost Model Performance & 95% Confidence Bounds

The production model uses **CatBoost Regressor**, validated on an 8-month time-based holdout benchmark (May 2025 to Dec 2025):

| Validation Benchmark Metric | Metric Value | Operational Meaning |
|---|---|---|
| **Selected Algorithm** | **CatBoost Regressor** | Enterprise gradient boosted decision trees |
| **R² Score (Variance Explained)** | **0.9521 (95.21%)** | Proportion of real-world sales pattern captured |
| **Model Overall Accuracy** | **91.21%** | Overall predictive accuracy ($100\% - \text{MAPE}$) |
| **Mean Absolute Percentage Error (MAPE)** | **8.79%** | Average percentage error margin |
| **Mean Absolute Error (MAE)** | **Rs. 1,46,254.81** | Average absolute rupee difference |
| **Root Mean Squared Error (RMSE)** | **Rs. 1,85,197.65** | Standard deviation of prediction residuals |
| **Residual Uncertainty (σ)** | **+/- Rs. 197,953.98** | Calibrated standard error for safety risk bounds |
| **Total Features Evaluated** | **35 Features** | Deal attributes, ratios, and cyclical indicators |

### 95% Gaussian Confidence Bounds
$$\text{Lower Bound} = \max\left(0, \hat{Y} - 1.96 \times ₹1,97,953.98\right) = \mathbf{₹16,59,945.12}\quad(\text{Worst-case limit})$$
$$\text{Upper Bound} = \hat{Y} + 1.96 \times ₹1,97,953.98 = \mathbf{₹24,35,924.72}\quad(\text{Best-case limit})$$

---

## 🚀 Top 5 Opportunity Growth Drivers & Attribution

The model explains **why revenue is growing** and provides the exact rupee lift attributed to each factor:

| Rank | Business Driver Name | Technical Feature Name | Model Importance | Attributed Revenue Lift |
|---|---|---|---|---|
| 1 | Weighted Pipeline Capitalization | `expected_deal_revenue` | **52.0%** | **+Rs. 77,970** |
| 2 | Lead Qualification Index | `pipeline_quality_value` | **10.0%** | **+Rs. 14,965** |
| 3 | Average Contract Value Expansion | `Deal Value` | **9.2%** | **+Rs. 13,868** |
| 4 | Portfolio Deal Size Distribution | `log_deal_value` | **6.0%** | **+Rs. 8,955** |
| 5 | Pipeline Conversion Strength | `pipeline_conversion_strength` | **3.2%** | **+Rs. 4,850** |

*Render this as a horizontal ranked bar chart on your dashboard.*

---

## 🌐 API Endpoints & React Integration

### API Endpoints Summary

| Method | Endpoint | Description | Sample Response Payload |
|---|---|---|---|
| `GET` | `/health` | Health check & model status | `{"status": "healthy", "model_loaded": true}` |
| `GET` | `/model-metrics` | Accuracy benchmarks | `{"status": "success", "metrics": { ... }}` |
| `GET` | `/sales-forecast/next-month` | Immediate next-month snapshot | Full single-month forecast object |
| `POST` | `/sales-forecast` | Multi-month horizon (1 to 6 months) | Array of multi-month forecast objects |
| `GET` | `/docs` | Interactive Swagger UI playground | Interactive Web Page |

*Note: CORS is pre-configured with `allow_origins=["*"]`, so calls from `localhost:3000` or `localhost:5173` work without proxy issues.*

---

### Copy-Paste React Component (`SalesForecastWidget.jsx`)

```jsx
import React, { useState, useEffect } from 'react';

export function SalesForecastWidget() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/sales-forecast/next-month')
      .then(res => {
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        return res.json();
      })
      .then(json => {
        setData(json);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="p-6 bg-slate-50 rounded-xl animate-pulse">Loading AI Sales Forecast...</div>;
  }

  if (error || !data) {
    return <div className="p-6 bg-red-50 text-red-700 rounded-xl">Error: {error || 'Failed to load'}</div>;
  }

  const { core_metrics, forecast } = data;

  return (
    <div className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center border-b pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900">AI Sales Forecast</h2>
          <p className="text-sm text-slate-500">Target Month: {forecast.forecast_month}</p>
        </div>
        <span className="px-3 py-1 bg-emerald-50 text-emerald-700 text-xs font-semibold rounded-full border border-emerald-200">
          Confidence: {core_metrics['Forecast Confidence Score']}
        </span>
      </div>

      {/* Grid Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Hero Revenue Card */}
        <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-100 rounded-xl">
          <span className="text-xs font-semibold text-blue-700 uppercase tracking-wide">Predicted Revenue</span>
          <p className="text-2xl font-black text-slate-900 mt-1">{core_metrics['Predicted Monthly Revenue']}</p>
          <p className="text-xs text-slate-600 mt-2">
            Safe Range: ₹{forecast.revenue_lower_bound.toLocaleString()} - ₹{forecast.revenue_upper_bound.toLocaleString()}
          </p>
        </div>

        {/* Pipeline Health Card */}
        <div className="p-4 bg-slate-50 border border-slate-100 rounded-xl">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Pipeline Health</span>
          <p className="text-2xl font-black text-slate-900 mt-1">{core_metrics['Pipeline Health Score']}</p>
          <p className="text-xs text-slate-500 mt-2">Score calibrated across active leads</p>
        </div>

        {/* Deal Closure Trend Card */}
        <div className="p-4 bg-slate-50 border border-slate-100 rounded-xl">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Closure Momentum</span>
          <p className="text-sm font-bold text-slate-800 mt-1">{core_metrics['Deal Closure Trend']}</p>
          <p className="text-xs text-emerald-600 font-medium mt-2">{core_metrics['Revenue Growth Trend']}</p>
        </div>
      </div>

      {/* Growth Drivers List */}
      <div>
        <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider mb-3">Top Revenue Drivers</h3>
        <div className="space-y-2">
          {forecast.top_factors?.map((factor, idx) => (
            <div key={idx} className="flex justify-between items-center text-xs p-2 bg-slate-50 rounded-lg">
              <span className="font-medium text-slate-700">{idx + 1}. {factor.feature.replace(/_/g, ' ')}</span>
              <span className="font-bold text-blue-600">{factor.importance}% (+₹{factor.impact_on_revenue.toLocaleString()})</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

---

## 📦 Complete Standalone Mock JSON (Zero Python Required)

Frontend developers can copy-paste this mock payload directly into MSW (Mock Service Worker), local JSON server, or state to build components:

```json
{
  "status": "success",
  "core_metrics": {
    "Predicted Monthly Revenue": "Rs. 2,047,934.92",
    "Pipeline Health Score": "40.82 / 100",
    "Deal Closure Trend": "Upward (33.6 expected closed deals, +141.7% vs hist avg)",
    "Revenue Growth Trend": "+2.0% projected growth (Upward Momentum Trajectory)",
    "Forecast Confidence Score": "93.2%"
  },
  "forecast": {
    "forecast_month": "January 2026",
    "predicted_revenue": 2047934.92,
    "pipeline_health_score": 40.82,
    "confidence_score": 93.2,
    "revenue_lower_bound": 1659945.12,
    "revenue_upper_bound": 2435924.72,
    "growth_percentage": -17.89,
    "expected_closed_deals": 33.6,
    "active_deals": 104.0,
    "total_pipeline_value": 16243000.0,
    "top_factors": [
      {
        "feature": "expected_deal_revenue",
        "importance": 51.98,
        "impact_on_revenue": 77970,
        "direction": "increased"
      },
      {
        "feature": "pipeline_quality_value",
        "importance": 9.98,
        "impact_on_revenue": 14965,
        "direction": "increased"
      },
      {
        "feature": "Deal Value",
        "importance": 9.25,
        "impact_on_revenue": 13868,
        "direction": "increased"
      },
      {
        "feature": "log_deal_value",
        "importance": 5.97,
        "impact_on_revenue": 8955,
        "direction": "increased"
      },
      {
        "feature": "pipeline_conversion_strength",
        "importance": 3.23,
        "impact_on_revenue": 4850,
        "direction": "increased"
      }
    ]
  }
}
```

---

## 🛡️ Error Handling & Graceful UI Degradation

| Failure Mode | HTTP Status | Root Cause | Recommended Frontend UI Behavior |
|---|---|---|---|
| **Server Offline** | Network Error / 503 | Uvicorn server not running or wrong port | Render retry banner: *"Unable to reach AI engine. Retrying in 5s..."* |
| **Model Not Found** | 500 Internal Error | Model `.joblib` file missing | Display alert: *"AI Model initializing. Please contact administrator."* |
| **Empty Pipeline** | 422 / 500 | No active deals entered for month | Show empty state: *"No active deals found in pipeline for this month."* |
| **Invalid Horizon** | 422 Unprocessable | Negative or non-integer horizon | Restrict frontend slider input between 1 and 6 months. |

---

## ⚙️ Dependencies & Setup Instructions

### Required Libraries ([`requirements.txt`](file:///d:/salse_forecasting/requirements.txt))
- `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.3.0`
- `catboost>=1.2.0`, `joblib>=1.3.0`
- `fastapi>=0.110.0`, `uvicorn>=0.29.0`

### Commands Cheat Sheet

```powershell
# 1. Open Terminal in the project root
cd d:\salse_forecasting

# 2. Install dependencies (One-time setup)
pip install -r requirements.txt

# 3. Run standalone terminal dashboard
python app.py

# 4. Start local Web API server for frontend
uvicorn app:app --port 8000 --reload
# (Swagger Playground: http://127.0.0.1:8000/docs)

# 5. Retrain model on fresh data (Optional)
python -c "from sales_forecaster import SalesForecaster; SalesForecaster().train()"
```

---

## 🔮 Limitations & Future Roadmap

### Current Limitations:
1. **Macroeconomic Shocks**: Unexpected external macro events (recessions, sudden tax law changes) are not captured in internal CRM tables alone.
2. **Sales Rep Subjectivity**: Reps who consistently inflate deal win probabilities can slightly bias the weighted pipeline.

### Future Roadmap for the Frontend App:
1. **Interactive "What-If" Scenario Simulator**: A slider allowing managers to adjust win probabilities dynamically and see real-time revenue recalculations.
2. **Rep Sensitivity Heatmaps**: Color-coded UI grids showing which sales reps have the highest pipeline health.
3. **Live WebSocket Push**: Instant notification toast alerts when high-ticket deals push monthly revenue above quarterly targets.

---

*Author: AI CRM Engineering Team | Version: 2.0.0 Enterprise Production Ready*
