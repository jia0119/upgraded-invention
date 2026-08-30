import numpy as np
import pandas as pd
import joblib
import streamlit as st
from datetime import date

st.set_page_config(page_title="Seoul Bike Demand Predictor", page_icon="🚲", layout="wide")

models = joblib.load("all_models.pkl")
comparison_df = joblib.load("comparison_df.pkl")

numerical_features = [
    "Hour", "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)",
    "Visibility (10m)", "Dew point temperature(°C)", "Solar Radiation (MJ/m2)",
    "Rainfall(mm)", "Snowfall (cm)", "DayOfWeek", "Month", "IsWeekend"
]
categorical_features = ["Seasons", "Holiday", "Functioning Day"]

best_model_name = comparison_df.sort_values("Test RMSE").iloc[0]["Model"]

st.title("🚲 Seoul Bike Sharing Demand Predictor")
st.markdown(
    "Predicts hourly bike rental demand using the four regression models built in "
    "Section 5.0 Modelling. Metrics shown are the actual Train/Test results computed "
    "in Section 6.0 Evaluation — not re-estimated live."
)

st.sidebar.header("⚙️ Model & Inputs")
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

st.sidebar.divider()
st.sidebar.subheader("📅 Date & Time")
input_date = st.sidebar.date_input("Date", value=date.today())
hour = st.sidebar.slider("Hour of Day", 0, 23, 18)
season = st.sidebar.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"], index=2)
holiday = st.sidebar.selectbox("Holiday", ["No Holiday", "Holiday"])

st.sidebar.divider()
st.sidebar.subheader("🌡️ Weather Conditions")
temperature = st.sidebar.slider("Temperature (°C)", -20.0, 40.0, 22.0, step=0.5)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 8.0, 2.0, step=0.1)
visibility = st.sidebar.slider("Visibility (10m)", 0, 2000, 1500, step=50)
dew_point = st.sidebar.slider("Dew Point Temperature (°C)", -30.0, 30.0, 10.0, step=0.5)
solar_radiation = st.sidebar.slider("Solar Radiation (MJ/m2)", 0.0, 4.0, 0.5, step=0.05)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=35.0, value=0.0, step=0.5)
snowfall = st.sidebar.number_input("Snowfall (cm)", min_value=0.0, max_value=9.0, value=0.0, step=0.5)


def build_row(target_hour):
    d = {
        "Hour": target_hour,
        "Temperature(°C)": temperature,
        "Humidity(%)": humidity,
        "Wind speed (m/s)": wind_speed,
        "Visibility (10m)": visibility,
        "Dew point temperature(°C)": dew_point,
        "Solar Radiation (MJ/m2)": solar_radiation,
        "Rainfall(mm)": rainfall,
        "Snowfall (cm)": snowfall,
        "DayOfWeek": input_date.weekday(),
        "Month": input_date.month,
        "IsWeekend": 1 if input_date.weekday() in [5, 6] else 0,
        "Seasons": season,
        "Holiday": holiday,
        "Functioning Day": "Yes"
    }
    return d


current_input_df = pd.DataFrame([build_row(hour)])[numerical_features + categorical_features]

best_r2 = comparison_df.loc[comparison_df["Model"] == best_model_name, "Test R2"].values[0]
st.success(
    f"🏆 **Recommended Best Model: {best_model_name}** | Highest Test R² Score: **{best_r2:.3f}**"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏆 Predictions", "📈 24-Hour Forecast", "📁 Batch Prediction (CSV)", "📊 Model Leaderboard"]
)

with tab1:
    st.subheader("Model Predictions")

    results = {}
    for name, model in models.items():
        pred = model.predict(current_input_df)[0]
        results[name] = max(0, int(round(pred)))

    col_best, col_selected = st.columns(2)
    with col_best:
        st.metric(f"🏆 Best Model ({best_model_name})", f"{results[best_model_name]:,} bikes")
    with col_selected:
        st.metric(f"Selected Model ({selected_model_name})", f"{results[selected_model_name]:,} bikes")

    st.divider()
    results_df = pd.DataFrame(list(results.items()), columns=["Model", "Predicted Bike Count"])
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        st.bar_chart(results_df.set_index("Model")["Predicted Bike Count"])
    with col_table:
        st.dataframe(results_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("24-Hour Forecast Curve")
    hours_list = list(range(24))
    daily_rows = pd.DataFrame([build_row(h) for h in hours_list])[numerical_features + categorical_features]

    trend_data = {"Hour": hours_list}
    for name, model in models.items():
        preds = model.predict(daily_rows)
        trend_data[name] = np.maximum(0, np.round(preds)).astype(int)

    trend_df = pd.DataFrame(trend_data).set_index("Hour")
    st.line_chart(trend_df)

with tab3:
    st.subheader("Batch CSV Prediction")
    st.markdown(
        "Upload a CSV with one row per prediction. Required columns "
        f"(exact names): `{', '.join(numerical_features + categorical_features)}`."
    )

    batch_model_name = st.selectbox(
        "Select Model for Batch Scoring",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name)
    )

    template_df = pd.DataFrame(columns=numerical_features + categorical_features)
    st.download_button(
        "📄 Download CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="batch_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        user_csv = pd.read_csv(uploaded_file)

        # If the raw dataset (with a Date column) was uploaded instead of pre-engineered
        # features, derive DayOfWeek/Month/IsWeekend automatically, same as Section 4.3.
        if "Date" in user_csv.columns and {"DayOfWeek", "Month", "IsWeekend"}.difference(user_csv.columns):
            parsed_date = pd.to_datetime(user_csv["Date"], dayfirst=True, errors="coerce")
            user_csv["DayOfWeek"] = parsed_date.dt.dayofweek
            user_csv["Month"] = parsed_date.dt.month
            user_csv["IsWeekend"] = user_csv["DayOfWeek"].isin([5, 6]).astype(int)

        required_cols = numerical_features + categorical_features
        missing_cols = [c for c in required_cols if c not in user_csv.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            batch_model = models[batch_model_name]
            batch_preds = batch_model.predict(user_csv[required_cols])
            user_csv[f"Predicted_Bikes_{batch_model_name}"] = np.maximum(0, np.round(batch_preds)).astype(int)

            st.success(f"Generated predictions using {batch_model_name}.")
            st.dataframe(user_csv, use_container_width=True)

            st.download_button(
                "📥 Download Scored CSV",
                data=user_csv.to_csv(index=False).encode("utf-8"),
                file_name="bike_predictions.csv",
                mime="text/csv"
            )

with tab4:
    st.subheader("Model Leaderboard (Real Computed Metrics — Section 6.0 Evaluation)")
    display_df = comparison_df.sort_values("Test RMSE").reset_index(drop=True)
    st.dataframe(
        display_df.style.highlight_max(subset=["Test R2"], color="#d4edda")
                         .highlight_min(subset=["Test RMSE", "Test MAE"], color="#d4edda"),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader(f"Feature Importance — {selected_model_name}")
    selected_model = models[selected_model_name]
    model_step = selected_model.named_steps["model"]

    if hasattr(model_step, "feature_importances_"):
        feature_names = selected_model.named_steps["preprocessor"].get_feature_names_out()
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model_step.feature_importances_
        }).sort_values("Importance", ascending=True).tail(10)
        st.bar_chart(fi_df.set_index("Feature"))
    else:
        st.info("Feature importance is not available for Linear Regression.")