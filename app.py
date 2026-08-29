import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(
    page_title="Seoul Bike Rental Analytics & Model Evaluation",
    page_icon="🚲",
    layout="wide",
)

FEATURE_NAMES = [
    "Hour",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "Seasons_Spring",
    "Seasons_Summer",
    "Seasons_Winter",
    "Holiday_No Holiday",
    "Functioning Day_Yes",
]


@st.cache_resource
def load_and_evaluate_models():
    """Train/load models and compute performance metrics to determine the best model."""
    np.random.seed(42)
    n_samples = 400
    X_dummy = pd.DataFrame(
        np.random.randn(n_samples, len(FEATURE_NAMES)), columns=FEATURE_NAMES
    )
    y_dummy = np.maximum(
        0,
        (X_dummy["Temperature(°C)"] * 35)
        + (X_dummy["Hour"] * 50)
        - (X_dummy["Humidity(%)"] * 10)
        + 600
        + np.random.normal(0, 40, size=n_samples),
    )

    candidate_models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=60, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Linear Regression": LinearRegression(),
    }

    models = {}
    metrics = {}

    for name, model in candidate_models.items():
        model.fit(X_dummy, y_dummy)
        preds = model.predict(X_dummy)

        r2 = r2_score(y_dummy, preds)
        rmse = np.sqrt(mean_squared_error(y_dummy, preds))
        mae = mean_absolute_error(y_dummy, preds)

        models[name] = model
        metrics[name] = {"R2 Score": r2, "RMSE": rmse, "MAE": mae}

    # Best model selection logic (highest R² Score)
    best_model_name = max(metrics, key=lambda k: metrics[k]["R2 Score"])

    return models, metrics, best_model_name


models, metrics, best_model_name = load_and_evaluate_models()

# Header
st.title("🚲 Seoul Bike Rental Demand & Model Recommendation Hub")
st.markdown(
    "Compare real-time predictions, evaluate model performance metrics, and automatically highlight the **Best Model**."
)

# Sidebar Inputs
st.sidebar.header("⚙️ Configuration & Inputs")
selected_model_name = st.sidebar.selectbox(
    "Select Model for Inspection", list(models.keys())
)

st.sidebar.divider()
st.sidebar.subheader("📅 Temporal Parameters")
hour = st.sidebar.slider("Hour of Day (0–23)", 0, 23, 17)
seasons = st.sidebar.selectbox(
    "Season", ["Spring", "Summer", "Autumn", "Winter"]
)
holiday = st.sidebar.selectbox("Holiday Status", ["No Holiday", "Holiday"])

st.sidebar.divider()
st.sidebar.subheader("🌡️ Weather Conditions")
temp = st.sidebar.slider("Temperature (°C)", -20.0, 40.0, 22.0, step=0.5)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 45)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 10.0, 1.8, step=0.1)
visibility = st.sidebar.slider("Visibility (10m)", 0, 2000, 1800, step=50)
dew_point = st.sidebar.slider(
    "Dew Point Temp (°C)", -35.0, 30.0, 9.5, step=0.5
)
solar_rad = st.sidebar.slider(
    "Solar Radiation (MJ/m²)", 0.0, 4.0, 1.2, step=0.05
)
rainfall = st.sidebar.number_input(
    "Rainfall (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5
)
snowfall = st.sidebar.number_input(
    "Snowfall (cm)", min_value=0.0, max_value=20.0, value=0.0, step=0.5
)


def build_feature_dict(target_hour):
    return {
        "Hour": target_hour,
        "Temperature(°C)": temp,
        "Humidity(%)": humidity,
        "Wind speed (m/s)": wind_speed,
        "Visibility (10m)": visibility,
        "Dew point temperature(°C)": dew_point,
        "Solar Radiation (MJ/m2)": solar_rad,
        "Rainfall(mm)": rainfall,
        "Snowfall (cm)": snowfall,
        "Seasons_Spring": 1 if seasons == "Spring" else 0,
        "Seasons_Summer": 1 if seasons == "Summer" else 0,
        "Seasons_Winter": 1 if seasons == "Winter" else 0,
        "Holiday_No Holiday": 1 if holiday == "No Holiday" else 0,
        "Functioning Day_Yes": 1,  # Fixed to 1 for continuous active predictions
    }


current_input_df = pd.DataFrame([build_feature_dict(hour)]).reindex(
    columns=FEATURE_NAMES, fill_value=0
)

# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏆 Best Model & Predictions",
        "📈 24-Hour Forecast Curve",
        "📁 Batch Prediction (CSV)",
        "🔍 Model Leaderboard & Insights",
    ]
)

# TAB 1: Best Model Recommendation & Multi-Model Comparison
with tab1:
    st.subheader("Model Predictions & Recommendation")

    # Predict across all models
    results = {}
    for name, model in models.items():
        pred = model.predict(current_input_df)[0]
        results[name] = max(0, int(round(pred)))

    best_prediction = results[best_model_name]

    # Top Banner: Best Model Prediction Recommendation
    st.success(
        f"🏆 **Recommended Best Model:** **{best_model_name}** | "
        f"Highest R² Score: **{metrics[best_model_name]['R2 Score']:.3f}**"
    )

    col_best, col_selected, col_ensemble = st.columns(3)

    with col_best:
        st.metric(
            label=f"🏆 Best Model ({best_model_name})",
            value=f"{best_prediction:,} bikes",
        )

    with col_selected:
        st.metric(
            label=f"Selected Model ({selected_model_name})",
            value=f"{results[selected_model_name]:,} bikes",
        )

    with col_ensemble:
        avg_val = int(np.mean(list(results.values())))
        st.metric(label="Ensemble Average", value=f"{avg_val:,} bikes")

    st.divider()

    # Side-by-Side Chart and Data Table
    col_chart, col_table = st.columns([2, 1])

    results_df = pd.DataFrame(
        list(results.items()), columns=["Model", "Predicted Bike Count"]
    )
    results_df["Is Best Model"] = results_df["Model"].apply(
        lambda x: "🏆 Best" if x == best_model_name else ""
    )

    with col_chart:
        st.subheader("Comparison Across All Models")
        st.bar_chart(
            results_df.set_index("Model")["Predicted Bike Count"],
            color="#29b5e8",
        )

    with col_table:
        st.subheader("Prediction Details")
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
        )

# TAB 2: 24-Hour Curve Simulation
with tab2:
    st.subheader("24-Hour Forecast Curve Comparison")

    hours_list = list(range(24))
    daily_records = [build_feature_dict(h) for h in hours_list]
    full_day_df = pd.DataFrame(daily_records).reindex(
        columns=FEATURE_NAMES, fill_value=0
    )

    trend_data = {"Hour": hours_list}
    for name, model in models.items():
        preds = model.predict(full_day_df)
        trend_data[name] = np.maximum(0, np.round(preds)).astype(int)

    trend_df = pd.DataFrame(trend_data).set_index("Hour")
    st.line_chart(trend_df)

# TAB 3: Batch CSV Scoring
with tab3:
    st.subheader("Batch CSV Prediction")
    batch_model_choice = st.selectbox(
        "Select Model for Batch CSV Scoring",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            user_csv = pd.read_csv(uploaded_file)
            processed_csv = user_csv.reindex(
                columns=FEATURE_NAMES, fill_value=0
            )

            # Force Functioning Day to 1 if omitted in uploaded CSV
            if "Functioning Day_Yes" not in user_csv.columns:
                processed_csv["Functioning Day_Yes"] = 1

            selected_m = models[batch_model_choice]
            raw_preds = selected_m.predict(processed_csv)
            user_csv[f"Predicted_Bikes_{batch_model_choice}"] = np.maximum(
                0, np.round(raw_preds)
            ).astype(int)

            st.success(
                f"Generated predictions using {batch_model_choice}!"
            )
            st.dataframe(user_csv.head(10))

            csv_export = user_csv.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored CSV",
                data=csv_export,
                file_name="bike_predictions.csv",
                mime="text/csv",
            )
        except Exception as err:
            st.error(f"Error processing CSV: {err}")

# TAB 4: Model Leaderboard & Feature Importance
with tab4:
    st.subheader("🏆 Model Leaderboard & Evaluation Metrics")
    st.markdown(
        "Models are evaluated on **R² Score** (higher is better), **RMSE** (lower is better), and **MAE** (lower is better)."
    )

    leaderboard_df = (
        pd.DataFrame.from_dict(metrics, orient="index")
        .reset_index()
        .rename(columns={"index": "Model"})
    )
    leaderboard_df = leaderboard_df.sort_values(
        by="R2 Score", ascending=False
    )

    st.dataframe(
        leaderboard_df.style.highlight_max(
            subset=["R2 Score"], color="#d4edda"
        ).highlight_min(subset=["RMSE", "MAE"], color="#d4edda"),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Feature Importance Analysis")
    if hasattr(models[best_model_name], "feature_importances_"):
        fi_df = pd.DataFrame(
            {
                "Feature": FEATURE_NAMES,
                "Importance": models[best_model_name].feature_importances_,
            }
        ).sort_values(by="Importance", ascending=True)

        st.caption(f"Feature importance breakdown for **{best_model_name}**:")
        st.bar_chart(fi_df.set_index("Feature"))
