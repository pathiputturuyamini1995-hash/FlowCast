from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from tensorflow.keras.models import load_model

# =========================================================
# FLOWCAST - AI-POWERED SMART TRAFFIC FLOW PREDICTION
# =========================================================

st.set_page_config(
    page_title="FlowCast | Smart Traffic Prediction",
    page_icon="🚦",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

TRAFFIC_FILE = DATA_DIR / "traffic_sensor_log.csv"
FEATURES_FILE = DATA_DIR / "processed" / "flowcast_features.csv"
ACCIDENT_DATA_FILE = FEATURES_FILE
RF_MODEL_FILE = MODELS_DIR / "traffic_volume_random_forest.pkl"
RF_FEATURES_FILE = MODELS_DIR / "traffic_volume_features.json"
CONGESTION_MODEL_FILE = MODELS_DIR / "congestion_classifier.pkl"
CONGESTION_FEATURES_FILE = MODELS_DIR / "congestion_features.json"
LSTM_MODEL_FILE = MODELS_DIR / "traffic_volume_lstm.keras"
LSTM_SCALER_FILE = MODELS_DIR / "traffic_volume_lstm_scaler.pkl"

# =========================================================
# LOAD TRAFFIC DATA
# =========================================================

@st.cache_data
def load_traffic_data():
    if not TRAFFIC_FILE.exists():
        raise FileNotFoundError(
            f"Traffic dataset not found: {TRAFFIC_FILE}"
        )

    df = pd.read_csv(TRAFFIC_FILE)
    df.columns = df.columns.str.strip()

    for column in ["date", "time", "road_name", "road_id"]:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    numeric_columns = [
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time",
        "accident_count",
        "road_capacity",
        "rainfall",
        "visibility",
        "temperature",
        "humidity",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )
    elif "date" in df.columns and "time" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str)
            + " "
            + df["time"].astype(str),
            errors="coerce",
        )
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["date"], errors="coerce"
        )

    return df
    
@st.cache_data
def load_feature_data():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Processed feature dataset not found: {FEATURES_FILE}"
        )

    feature_df = pd.read_csv(FEATURES_FILE)
    feature_df.columns = feature_df.columns.str.strip()

    if "timestamp" in feature_df.columns:
        feature_df["timestamp"] = pd.to_datetime(
            feature_df["timestamp"], errors="coerce"
        )

    return feature_df


# =========================================================
# LOAD SAVED RANDOM FOREST MODEL AND FEATURES
# =========================================================


@st.cache_resource
def load_lstm_model():
    if not LSTM_MODEL_FILE.exists():
        return None
    return load_model(str(LSTM_MODEL_FILE))


@st.cache_resource
def load_lstm_scaler():
    if not LSTM_SCALER_FILE.exists():
        return None
    return joblib.load(LSTM_SCALER_FILE)

    
@st.cache_resource
def load_rf_model():
    if not RF_MODEL_FILE.exists():
        return None

    try:
        return joblib.load(RF_MODEL_FILE)
    except Exception:
        return None


@st.cache_data
def load_rf_features():
    if not RF_FEATURES_FILE.exists():
        return []

    try:
        with open(
            RF_FEATURES_FILE, "r", encoding="utf-8"
        ) as file:
            saved_features = json.load(file)

        if isinstance(saved_features, list):
            return saved_features

        if isinstance(saved_features, dict):
            for key in [
                "features",
                "feature_names",
                "columns",
                "selected_features",
            ]:
                if isinstance(saved_features.get(key), list):
                    return saved_features[key]

        return []

    except (OSError, json.JSONDecodeError):
        return []

@st.cache_resource
def load_congestion_model():
    model = joblib.load(CONGESTION_MODEL_FILE)

    with open(CONGESTION_FEATURES_FILE, "r", encoding="utf-8") as f:
        features = json.load(f)

    return model, features

@st.cache_resource
def load_accident_model():
    model_path = MODELS_DIR / "accident_risk_60m_extra_trees.joblib"
    bundle = joblib.load(model_path)

    model = bundle["model"]
    imputer = bundle["imputer"]
    features = bundle["features"]
    threshold = bundle.get("threshold", 0.5)

    return model, imputer, features, threshold
# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_features(history):
    result = history.copy()

    group_col = "road_id" if "road_id" in result.columns else "road_name"
    result = result.sort_values([group_col, "timestamp"]).copy()

    result["hour"] = result["timestamp"].dt.hour
    result["minute"] = result["timestamp"].dt.minute
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["month"] = result["timestamp"].dt.month

    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)

    result["is_morning_peak"] = (
        result["hour"].between(7, 10)
    ).astype(int)

    result["is_evening_peak"] = (
        result["hour"].between(16, 19)
    ).astype(int)

    result["traffic_volume_lag_1"] = (
    result.groupby(group_col)["traffic_volume"].shift(1)
    )

    result["traffic_volume_lag_2"] = (
    result.groupby(group_col)["traffic_volume"].shift(2)
    )

    result["traffic_volume_lag_4"] = (
    result.groupby(group_col)["traffic_volume"].shift(4)
    )

    result["traffic_volume_rolling_mean_4"] = (
    result.groupby(group_col)["traffic_volume"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    )

    if "avg_speed" in result.columns:
        result["avg_speed_lag_1"] = (
            result.groupby(group_col)["avg_speed"].shift(1)
        )

        result["avg_speed_lag_2"] = (
            result.groupby(group_col)["avg_speed"].shift(2)
        )

        result["avg_speed_rolling_mean_4"] = (
            result.groupby(group_col)["avg_speed"]
            .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        )

    return result
# =========================================================
# PAGE HEADER AND INITIALIZATION
# =========================================================

st.title("🚦 FlowCast")
st.subheader(
    "AI-Powered Smart Traffic Flow Prediction Platform"
)
st.write(
    "Monitor traffic conditions, explore congestion patterns, "
    "and review traffic sensor data."
)

rf_model = load_rf_model()
rf_features = load_rf_features()

if rf_model is not None:
    st.sidebar.success("Random Forest model file loaded")
else:
    st.sidebar.warning("Random Forest model unavailable")


# =========================================================
# MAIN DASHBOARD
# =========================================================

try:
    df = load_traffic_data()

    if df.empty:
        st.warning("The traffic dataset is empty.")
        st.stop()

    st.sidebar.header("Navigation")

    section = st.sidebar.radio(
        "Choose a section",
        [
            "Dashboard Overview",
            "Live Predictions",
            "Traffic Analysis",
            "Congestion Analysis",
            "Weather vs Traffic",
            "Accident Risk",
            "Data Explorer",
        ],
    )

    # =====================================================
    # 1. DASHBOARD OVERVIEW
    # =====================================================

    if section == "Dashboard Overview":

        st.header("Traffic Dashboard Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Traffic records",
            f"{len(df):,}",
        )

        if "road_id" in df.columns:
            road_count = df["road_id"].nunique()
        elif "road_name" in df.columns:
            road_count = df["road_name"].nunique()
        else:
            road_count = None

        if road_count is not None:
            c2.metric(
                "Road segments",
                f"{road_count:,}",
            )

        if "traffic_volume" in df.columns:
            avg_volume = df["traffic_volume"].mean()
            if pd.notna(avg_volume):
                c3.metric(
                    "Average traffic volume",
                    f"{avg_volume:,.1f}",
                )

        if "avg_speed" in df.columns:
            avg_speed = df["avg_speed"].mean()
            if pd.notna(avg_speed):
                c4.metric(
                    "Average speed",
                    f"{avg_speed:.1f} km/h",
                )

        st.info(
            "These statistics describe historical records, "
            "not live traffic measurements."
        )

        if (
            "timestamp" in df.columns
            and "traffic_volume" in df.columns
        ):
            trend = (
                df.dropna(
                    subset=["timestamp", "traffic_volume"]
                )
                .groupby("timestamp", as_index=False)
                ["traffic_volume"]
                .mean()
                .sort_values("timestamp")
            )

            st.subheader("Traffic Volume Over Time")

            st.plotly_chart(
                px.line(
                    trend,
                    x="timestamp",
                    y="traffic_volume",
                    title="Observed Traffic Volume",
                ),
                use_container_width=True,
            )

        if "congestion_level" in df.columns:
            st.subheader("Congestion Level Distribution")

            counts = (
                df["congestion_level"]
                .fillna("Unknown")
                .replace("", "Unknown")
                .value_counts()
                .rename_axis("Congestion Level")
                .reset_index(name="Records")
            )

            st.plotly_chart(
                px.bar(
                    counts,
                    x="Congestion Level",
                    y="Records",
                ),
                use_container_width=True,
            )

    # =====================================================
    # 2. TRAFFIC ANALYSIS
    # =====================================================

    elif section == "Traffic Analysis":

        st.header("Traffic Analysis")

        view = df.copy()

        if "road_name" in df.columns:
            road_names = sorted(
                df["road_name"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            roads = ["All roads"] + road_names

            selected_road = st.selectbox(
                "Select road",
                roads,
            )

            if selected_road != "All roads":
                view = view[
                    view["road_name"].astype(str)
                    == selected_road
                ].copy()
        # Travel Time Analysis
        st.subheader("Travel Time Analysis")

        if "travel_time" in view.columns:

            travel_values = pd.to_numeric(
                view["travel_time"], errors="coerce"
            ).dropna()

            if not travel_values.empty:

                st.metric(
                    "Average Recorded Travel Time",
                    f"{travel_values.mean():.2f} minutes"
                )

                st.caption(
                    "Based on historical recorded segment travel time."
                )

                travel_df = view.dropna(
                    subset=["timestamp", "travel_time"]
                )

                if not travel_df.empty:

                    fig = px.line(
                        travel_df.sort_values("timestamp"),
                        x="timestamp",
                        y="travel_time",
                        title="Travel Time Trend by Road"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:
                    st.info("No valid travel time records available.")

            else:
                st.info("Travel time values are unavailable.")

        else:
            st.warning("Travel time data is unavailable.")

        if (
            "timestamp" in view.columns
            and "traffic_volume" in view.columns
        ):
            trend = (
                view.dropna(
                    subset=["timestamp", "traffic_volume"]
                )
                .groupby("timestamp", as_index=False)
                ["traffic_volume"]
                .mean()
                .sort_values("timestamp")
            )

            st.plotly_chart(
                px.line(
                    trend,
                    x="timestamp",
                    y="traffic_volume",
                    title="Observed Traffic Volume by Time",
                ),
                use_container_width=True,
            )
            st.subheader("Actual vs Predicted Traffic Volume")

            if "traffic_volume" in view.columns and not view.empty:
                try:
                    model = load_rf_model()
                    feature_names = load_rf_features()

                    if model is None or not feature_names:
                        st.warning("Model or feature list is unavailable.")
                    else:
                        engineered = create_features(
                            view.sort_values("timestamp").copy()
                        )
                        engineered = engineered.dropna(
                            subset=feature_names + ["traffic_volume"]
                        )

                        if engineered.empty:
                            st.warning("Not enough data to make predictions.")
                        else:
                            chart_data = engineered.tail(200).copy()
                            chart_data["Predicted Traffic Volume"] = model.predict(
                                chart_data[feature_names]
                            )

                            chart_data = chart_data.rename(
                                columns={"traffic_volume": "Actual Traffic Volume"}
                            )

                            # Add empirical prediction confidence band
                            import plotly.graph_objects as go

                            error_margin = 115.28

                            chart_data["Lower_Bound"] = (
                                chart_data["Predicted Traffic Volume"] - error_margin
                            ).clip(lower=0)

                            chart_data["Upper_Bound"] = (
                                chart_data["Predicted Traffic Volume"] + error_margin
                            )

                            fig = go.Figure()

                            # Lower boundary
                            fig.add_trace(go.Scatter(
                                x=chart_data["timestamp"],
                                y=chart_data["Lower_Bound"],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False
                            ))

                            # Upper boundary with shaded confidence band
                            fig.add_trace(go.Scatter(
                                x=chart_data["timestamp"],
                                y=chart_data["Upper_Bound"],
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor="rgba(100,149,237,0.20)",
                                name="Empirical Prediction Interval"
                             ))

                            # Actual traffic volume
                            fig.add_trace(go.Scatter(
                                x=chart_data["timestamp"],
                                y=chart_data["Actual Traffic Volume"],
                                mode="lines",
                                name="Actual Traffic Volume"
                            ))

                            # Predicted traffic volume
                            fig.add_trace(go.Scatter(
                                x=chart_data["timestamp"],
                                y=chart_data["Predicted Traffic Volume"],
                                mode="lines",
                                name="Predicted Traffic Volume"
                            ))

                            fig.update_layout(
                                title="Actual vs Predicted Traffic Volume with Prediction Interval",
                                xaxis_title="Timestamp",
                                yaxis_title="Traffic Volume",
                                hovermode="x unified"
                            )

                            st.plotly_chart(fig, use_container_width=True)
                            st.subheader("Download Traffic Prediction Report")

                            report_data = chart_data[
                                [
                                    "timestamp",
                                    "Actual Traffic Volume",
                                    "Predicted Traffic Volume",
                                    "Lower_Bound",
                                    "Upper_Bound"
                                ]
                            ].copy()

                            csv_report = report_data.to_csv(index=False).encode("utf-8")

                            st.download_button(
                                label="Download Traffic Report (CSV)",
                                data=csv_report,
                            file_name="FlowCast_Traffic_Prediction_Report.csv",
                                mime="text/csv"
                            )
                except Exception as e:
                    st.error(f"Prediction chart error: {e}")
            else:
                st.info("Traffic volume data is unavailable for this selection.")

        st.dataframe(
            view.head(1000),
            use_container_width=True,
        )
        st.caption(
             "Evaluation results from the saved traffic volume "
             "model evaluation notebook."
        )
        st.subheader("Random Forest Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", "22.96")
        col2.metric("RMSE", "37.51")
        col3.metric("MAPE", "5.67%")

        # ---------------------------------------------
        # Random Forest Feature Importance
        # ---------------------------------------------

        st.subheader("Model Feature Importance")

        if (
            rf_model is not None
            and rf_features
            and hasattr(rf_model, "feature_importances_")
        ):
            importance_df = pd.DataFrame({
                "Feature": rf_features,
                "Importance": rf_model.feature_importances_
            })

            importance_df = importance_df.sort_values(
                "Importance", ascending=False
            ).head(10)

            fig = px.bar(
                importance_df.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 10 Traffic Prediction Features"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
        else:
            st.warning("Feature importance is unavailable.")
            
        # ---------------------------------------------
        # Traffic volume prediction
        # ---------------------------------------------

        st.subheader("Traffic Volume Prediction")

        if rf_model is None:
            st.warning(
                "The saved Random Forest model could not be loaded."
            )

        elif not rf_features:
            st.warning(
                "The model feature list could not be loaded. "
                "Check traffic_volume_features.json."
            )

        elif (
            "traffic_volume" not in view.columns
            or "timestamp" not in view.columns
        ):
            st.warning(
                "Timestamp or traffic volume column is missing."
            )

        else:
            road_history = (
                view.dropna(
                    subset=["timestamp", "traffic_volume"]
                )
                .sort_values("timestamp")
                .copy()
            )

            if len(road_history) < 5:
                st.warning(
                    "Not enough historical readings."
                )

            else:
                engineered = create_features(road_history)
                latest = engineered.iloc[-1]

                missing_features = [
                    feature
                    for feature in rf_features
                    if feature not in engineered.columns
                ]

                if missing_features:
                    st.warning(
                        "Missing required model features: "
                        + ", ".join(missing_features)
                    )

                else:
                    input_data = pd.DataFrame(
                        [[latest[feature] for feature in rf_features]],
                        columns=rf_features,
                    )

                    input_data = input_data.apply(
                        pd.to_numeric,
                        errors="coerce",
                    )

                    if input_data.isna().any().any():
                        st.warning(
                            "Some model input features are missing. "
                            "Check the data and feature engineering."
                        )

                    else:
                        try:
                            prediction = float(
                                rf_model.predict(input_data)[0]
                            )

                            st.metric(
                                "Predicted Traffic Volume",
                                f"{max(0.0, prediction):,.0f}",
                            )

                            st.caption(
                                "Random Forest estimate based on "
                                "historical readings; not a live "
                                "traffic measurement."
                            )

                        except Exception as error:
                            st.error(
                                f"Prediction failed: {error}"
                            )
        # ============================================================
        # LSTM TRAFFIC FORECASTING
        # ============================================================

        st.subheader("LSTM Traffic Volume Forecast")

        lstm_model = load_lstm_model()
        lstm_scaler = load_lstm_scaler()

        if lstm_model is None or lstm_scaler is None:

            st.warning(
                "LSTM model or scaler not found. "
                "Check the files in the models folder."
            )

        elif "traffic_volume" not in view.columns:

            st.warning("Traffic volume data is unavailable.")

        else:

            try:

                lstm_data = (
                    view.sort_values("timestamp")
                    [["traffic_volume"]]
                    .dropna()
                    .astype(float)
                    .values
                    .reshape(-1, 1)
                )

                if len(lstm_data) < 12:

                    st.warning(
                        "At least 12 traffic records are required."
                    )

                else:

                    scaled_data = lstm_scaler.transform(lstm_data)

                    sequence = scaled_data[-12:].reshape(1, 12, 1)

                    forecast_steps = st.slider(
                        "Forecast steps",
                        min_value=1,
                        max_value=12,
                        value=6
                    )

                    predictions = []

                    for _ in range(forecast_steps):

                        next_scaled = lstm_model.predict(
                            sequence,
                            verbose=0
                        )

                        next_value = float(
                            next_scaled.reshape(-1)[0]
                        )

                        predictions.append(next_value)

                        next_sequence = np.append(
                            sequence.reshape(-1)[1:],
                            next_value
                        )

                        sequence = next_sequence.reshape(1, 12, 1)

                    predicted_values = lstm_scaler.inverse_transform(
                        np.array(predictions).reshape(-1, 1)
                    ).flatten()

                    last_timestamp = pd.to_datetime(
                      view["timestamp"], errors="coerce"
                    ).max()

                    if pd.isna(last_timestamp):
                       st.error("Unable to generate forecast: timestamp is missing.")
                       st.stop()

                    future_timestamps = [
                      last_timestamp + pd.Timedelta(minutes=30 * i)
                      for i in range(1, forecast_steps + 1)
                    ]

                    forecast_df = pd.DataFrame({
                      "Forecast Time": future_timestamps,
                      "Predicted Traffic Volume": np.maximum(
                        predicted_values, 0
                       ).round(0).astype(int)
                    })
                    # Add approximate prediction uncertainty range
                    LSTM_VALIDATION_MAE = 51.76

                    forecast_df["Lower Bound"] = np.maximum(
                      forecast_df["Predicted Traffic Volume"] - LSTM_VALIDATION_MAE, 0
                      ).round(0).astype(int)

                    forecast_df["Upper Bound"] = (
                      forecast_df["Predicted Traffic Volume"] + LSTM_VALIDATION_MAE
                      ).round(0).astype(int)
                        

                    st.dataframe(
                        forecast_df,
                        use_container_width=True
                    )
                    st.download_button(
                        label="Download LSTM Forecast Report",
                        data=forecast_df.to_csv(index=False),
                    file_name="flowcast_lstm_forecast_report.csv",
                        mime="text/csv"
                    )

                    fig = px.line(
                        forecast_df,
                        x="Forecast Time",
                        y="Predicted Traffic Volume",
                        markers=True,
                        title="LSTM Traffic Forecast with Approximate Uncertainty Range"
                    )

                    fig.add_scatter(
                        x=forecast_df["Forecast Time"],
                        y=forecast_df["Upper Bound"],
                        mode="lines",
                        line=dict(dash="dash"),
                        name="Upper Bound"
                    )

                    fig.add_scatter(
                        x=forecast_df["Forecast Time"],
                        y=forecast_df["Lower Bound"],
                        mode="lines",
                        line=dict(dash="dash"),
                        name="Lower Bound"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.caption(
                        "Forecasts are iterative predictions based on "
                        "the last 12 traffic-volume observations. "
                        "Each forecast step represents one model time step."
                    )

            except Exception as e:

                st.error(f"LSTM forecasting error: {e}")

        # ---------------------------------------------
        # Road Comparison
        # ---------------------------------------------

        st.subheader("Road Comparison")

        comparison_df = (
            df.groupby("road_name")
            .agg(
                Average_Traffic=("traffic_volume", "mean"),
                Average_Speed=("avg_speed", "mean")
            )
            .reset_index()
            .sort_values("Average_Traffic", ascending=False)
        )

        fig = px.bar(
            comparison_df,
            x="road_name",
            y="Average_Traffic",
            title="Average Traffic Volume by Road",
            labels={
                "road_name": "Road",
                "Average_Traffic": "Average Traffic Volume"
            }
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comparison_df, use_container_width=True)

    
   # =====================================================
    # 3. CONGESTION ANALYSIS
    # =====================================================

    elif section == "Congestion Analysis":

        st.subheader("Congestion Heatmap by Road and Time")

        heatmap_df = df.copy()

        if "date" in heatmap_df.columns and "time" in heatmap_df.columns:

            heatmap_df["timestamp"] = pd.to_datetime(
                heatmap_df["date"].astype(str) + " " +
                heatmap_df["time"].astype(str),
                errors="coerce"
            )

            severity_map = {
                "Free-Flow": 0,
                "Free-flow": 0,
                "Moderate": 1,
                "Heavy": 2,
                "Severe": 3
            }

            heatmap_df["severity"] = (
                heatmap_df["congestion_level"]
                .map(severity_map)
            )

            heatmap_df = heatmap_df.dropna(
                subset=["timestamp", "road_name", "severity"]
            )

            heatmap_df["time_window"] = (
                heatmap_df["timestamp"].dt.strftime("%m-%d %H:%M")
            )

            pivot = heatmap_df.pivot_table(
                index="road_name",
                columns="time_window",
                values="severity",
                aggfunc="mean"
            )

            if not pivot.empty:

                fig = px.imshow(
                    pivot,
                    labels={
                        "x": "Time",
                        "y": "Road",
                        "color": "Congestion"
                    },
                    zmin=0,
                    zmax=3,
                    aspect="auto"
                )

                fig.update_layout(
                    height=600,
                    xaxis_nticks=12,
                    yaxis_title="Road Name",
                    xaxis_title="Time",
                    title="Congestion Intensity by Road and Time"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.info("No congestion data available for the heatmap.")

        st.header("Congestion Analysis")

        # Historical congestion distribution

        if "congestion_level" in df.columns:

            counts = (
                df["congestion_level"]
                .fillna("Unknown")
                .replace("", "Unknown")
                .value_counts()
                .rename_axis("Congestion Level")
                .reset_index(name="Records")
            )

            st.plotly_chart(
                px.bar(
                    counts,
                    x="Congestion Level",
                    y="Records",
                    title="Recorded Congestion Levels"
                ),
                use_container_width=True
            )

            # Historical congestion by road

            if "road_name" in df.columns:

                st.subheader("Congestion by Road")

                congestion_by_road = (
                    df.assign(
                        congestion_level=(
                            df["congestion_level"]
                            .fillna("Unknown")
                            .replace("", "Unknown")
                        )
                    )
                    .groupby(
                        ["road_name", "congestion_level"],
                        dropna=False
                    )
                    .size()
                    .reset_index(name="Records")
                )

                st.dataframe(
                    congestion_by_road,
                    use_container_width=True
                )

        else:
            st.warning(
                "The congestion_level column was not found."
            )

        # Saved congestion model and predictions

        st.subheader("AI Congestion Predictions")

        try:

            congestion_model, congestion_features = (
                load_congestion_model()
            )

            st.success(
                "Congestion classifier loaded successfully."
            )
            st.subheader("Congestion Model Performance")

            col1, col2 = st.columns(2)

            col1.metric("Accuracy", "95.60%")
            col2.metric("Macro F1-Score", "0.9394")

            st.caption("Evaluation results from the saved congestion classification notebook.")

            feature_df = load_feature_data()

            # Check required features

            missing_features = [
                col for col in congestion_features
                if col not in feature_df.columns
            ]

            if missing_features:

                st.error(
                    f"Missing model features: {missing_features}"
                )

            else:

                # Prepare input in the exact training order

                X_congestion = feature_df.reindex(
                    columns=congestion_features
                )

                X_congestion = X_congestion.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                # Predict only rows with complete input features

                valid_rows = X_congestion.notna().all(axis=1)

                if valid_rows.sum() == 0:

                    st.warning(
                        "No complete rows available for congestion prediction."
                    )

                else:

                    predictions = congestion_model.predict(
                        X_congestion.loc[valid_rows]
                    )

                    congestion_results = (
                        feature_df.loc[valid_rows].copy()
                    )

                    congestion_results["Predicted Congestion"] = (
                        predictions
                    )

                    st.write(
                        "Records predicted:",
                        len(congestion_results)
                    )

                    if (~valid_rows).sum() > 0:
                        st.caption(
                            f"Skipped {(~valid_rows).sum()} rows "
                            "with missing input features."
                        )

                    # Sort by timestamp

                    if "timestamp" in congestion_results.columns:

                        congestion_results["timestamp"] = (
                            pd.to_datetime(
                                congestion_results["timestamp"],
                                errors="coerce"
                            )
                        )

                        congestion_results = (
                            congestion_results.sort_values("timestamp")
                        )

                    # Latest prediction for each road

                    if "road_id" in congestion_results.columns:

                        latest_results = (
                            congestion_results.drop_duplicates(
                                subset="road_id",
                                keep="last"
                            )
                        )
                    if "Predicted Congestion" not in latest_results.columns:
                        st.warning("Predicted Congestion column was not created.")
                    else:

                        latest_results = (
                            congestion_results.tail(25)
                        )

                    st.subheader("Latest Congestion Prediction by Road")

                    display_cols = [
                        col for col in [
                            "road_id",
                            "road_name",
                            "timestamp",
                            "traffic_volume",
                            "avg_speed",
                            "Predicted Congestion"
                        ]
                        if col in latest_results.columns
                    ]

                    st.dataframe(
                        latest_results[display_cols],
                        use_container_width=True
                    )

        except Exception as error:

            st.error(
                f"Congestion prediction error: {error}"
            )
    # =====================================================
    # WEATHER VS TRAFFIC ANALYSIS
    # =====================================================

    elif section == "Weather vs Traffic":

        st.header("Weather vs Traffic Analysis")
        
        feature_df = load_feature_data()
        weather_columns = [
            "temperature",
            "rainfall",
            "humidity",
            "visibility"
        ]

        available_weather = [
          col for col in weather_columns
          if col in feature_df.columns
        ]

        if "traffic_volume" not in feature_df.columns:
            st.warning("Traffic volume data is unavailable.")

        elif not available_weather:
            st.warning("Weather data is unavailable.")

        else:
            selected_weather = st.selectbox(
                "Select weather condition",
                available_weather
            )

            weather_df = feature_df[
             [selected_weather, "traffic_volume"]
            ].copy()

            weather_df[selected_weather] = pd.to_numeric(
                weather_df[selected_weather],
                errors="coerce"
            )

            weather_df["traffic_volume"] = pd.to_numeric(
                weather_df["traffic_volume"],
                errors="coerce"
            )

            weather_df = weather_df.dropna()

            if not weather_df.empty:

                st.subheader(
                    f"{selected_weather.title()} vs Traffic Volume"
                )

                fig = px.scatter(
                    weather_df,
                    x=selected_weather,
                    y="traffic_volume",
                    trendline="ols",
                    title=f"{selected_weather.title()} vs Traffic Volume"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                correlation = weather_df[
                    selected_weather
                ].corr(weather_df["traffic_volume"])

                st.metric(
                    "Weather and Traffic Correlation",
                    f"{correlation:.2f}"
                )

                st.caption(
                    "Correlation shows the relationship in historical "
                    "data; it does not prove that weather causes traffic changes."
                )

            else:
                st.warning("No valid weather records available.")       
            
    # =====================================================
    # 4. ACCIDENT RISK
    # =====================================================

    elif section == "Accident Risk":

        st.header("Accident Risk Analysis")

        if "accident_count" in df.columns:

            accident_total = (
                pd.to_numeric(
                    df["accident_count"],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )

            st.metric(
                "Recorded accidents",
                f"{accident_total:,.0f}",
            )

            st.caption(
                "Historical accident counts are shown here. "
                "They are not model-generated risk probabilities."
            )

            if "road_name" in df.columns:
                risk_view = df.copy()
                risk_view["accident_count"] = pd.to_numeric(
                    risk_view["accident_count"],
                    errors="coerce",
                ).fillna(0)

                risk_view = (
                    risk_view.groupby(
                        "road_name",
                        as_index=False,
                    )["accident_count"]
                    .sum()
                    .sort_values(
                        "accident_count",
                        ascending=False,
                    )
                )

                st.plotly_chart(
                    px.bar(
                        risk_view,
                        x="road_name",
                        y="accident_count",
                        title="Recorded Accidents by Road",
                    ),
                    use_container_width=True,
                )

        else:
            st.warning(
                "The accident_count column was not found."
            )

        st.subheader("Accident Risk Model Check")
        
        st.subheader("Accident Model Performance")

        col1, col2 = st.columns(2)

        col1.metric("ROC-AUC Score", "0.5779")
        col2.metric("PRD Target", "0.75")

        st.warning(
          "The current accident model is below the PRD target "
          "and requires further improvement."
        )

    

        try:
            (
                accident_model,
                accident_imputer,
                accident_features,
                accident_threshold,
            ) = load_accident_model()

            st.success("Accident Risk model loaded successfully!")
            st.write("Expected model features:", len(accident_features))

            feature_df = load_feature_data()

            missing_features = [
                col for col in accident_features
                if col not in feature_df.columns
            ]
            st.write(
                "Available features:",
                len(accident_features) - len(missing_features)
            )
            st.write("Missing features:", missing_features)
            
            if not missing_features:
                st.subheader("60-Minute Accident Risk Prediction")

                X_accident = feature_df.reindex(columns=accident_features)
                X_accident = X_accident.apply(
                    pd.to_numeric, errors="coerce"
                )

                X_accident_imputed = accident_imputer.transform(
                    X_accident
                )

                risk_probabilities = accident_model.predict_proba(
                    X_accident_imputed
                )[:, 1]
                st.info(
                    "Prediction reliability: Experimental. "
                    "These probabilities are model-generated risk estimates, "
                    "not calibrated confidence intervals. "
                    "The model has not yet achieved the PRD ROC-AUC target of 0.75."
                )
                st.write("Risk probability sample:", risk_probabilities[:10])

                risk_results = feature_df.copy()
                risk_results["Accident Risk Probability"] = (
                    risk_probabilities * 100
                ).round(2)

                risk_results["Predicted Risk"] = [
                    "High Risk" if p >= accident_threshold
                    else "Low Risk"
                    for p in risk_probabilities
                ]

                if "timestamp" in risk_results.columns:
                    risk_results["timestamp"] = pd.to_datetime(
                        risk_results["timestamp"], errors="coerce"
                    )
                    risk_results = risk_results.sort_values("timestamp")

                if "road_name" in risk_results.columns:
                    risk_results = risk_results.drop_duplicates(
                        subset="road_name", keep="last"
                    )
                # Accident risk probability chart
                if not risk_results.empty:
                   fig = px.bar(
                     risk_results.head(25),
                     x="road_name",
                     y="Accident Risk Probability",
                     color="Predicted Risk",
                     title="60-Minute Accident Risk by Road",
                     labels={
                       "road_name": "Road Name",
                       "Accident Risk Probability": "Risk Probability (%)"
                     }
                   )

                   st.plotly_chart(fig, use_container_width=True)
                else:
                   st.warning("No accident risk predictions available.")

                display_columns = [
                    col for col in [
                        "road_name",
                        "timestamp",
                        "Accident Risk Probability",
                        "Predicted Risk"
                    ]
                    if col in risk_results.columns
                ]

                st.dataframe(
                    risk_results[display_columns],
                    use_container_width=True
                )
            else:
                st.warning(
                    "Prediction unavailable. Required features are missing."
                )

        except Exception as e:
            st.error(f"Accident model error: {e}")

    # ==========================================
    # 5. DATA EXPLORER
    # ==========================================

    elif section == "Data Explorer":

        st.header("Traffic Data Explorer")

        st.write("Dataset shape:", df.shape)

        st.subheader("Missing Values by Column")

        missing_values = pd.DataFrame({
            "Column": df.columns,
            "Missing values": df.isnull().sum().values
        })

        st.dataframe(
            missing_values,
            use_container_width=True
        )
        # Download Data Quality Report
        quality_report_path = DATA_DIR / "processed" / "data_quality_report.csv"

        st.subheader("Download Data Quality Report")

        if quality_report_path.exists():
           quality_report_df = pd.read_csv(quality_report_path)

           st.download_button(
               label="Download Data Quality Report (CSV)",
           data=quality_report_df.to_csv(index=False).encode("utf-8"),
           file_name="FlowCast_Data_Quality_Report.csv",
                   mime="text/csv",
                   key="download_quality_report"
               )
        else:
            st.warning("Data quality report file not found.")

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(1000),
            use_container_width=True
        )
        st.divider()
        st.subheader("Upload New Traffic Data")

        uploaded_file = st.file_uploader(
            "Upload a traffic CSV file",
            type=["csv"],
            key="traffic_csv_upload"
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)

                if uploaded_df.empty:
                    st.error("The uploaded CSV is empty.")
                else:
                    st.success("CSV uploaded successfully!")
                    st.write("Uploaded dataset shape:", uploaded_df.shape)

                    # Create timestamp from date and time columns
                    if "timestamp" not in uploaded_df.columns:
                        if "date" in uploaded_df.columns and "time" in uploaded_df.columns:
                            uploaded_df["timestamp"] = pd.to_datetime(
                    uploaded_df["date"].astype(str) + " " +
                    uploaded_df["time"].astype(str),
                                errors="coerce"
                            )

                    required = {"timestamp", "traffic_volume"}
                    missing = required - set(uploaded_df.columns)

                    if not ({"road_id", "road_name"} & set(uploaded_df.columns)):
                        missing.add("road_id or road_name")

                    if missing:
                        st.error(f"Missing required columns: {sorted(missing)}")
                    else:
                        st.success("Basic column validation passed.")

                    st.write("Missing values:")
                    st.dataframe(uploaded_df.isnull().sum())

                    st.write("Uploaded data preview:")
                    st.dataframe(uploaded_df.head(20))
                    uploaded_path = DATA_DIR / "uploaded_traffic_data.csv"
                    uploaded_df.to_csv(uploaded_path, index=False)
                    # Clean uploaded traffic data
                    uploaded_df["timestamp"] = pd.to_datetime(
                        uploaded_df["timestamp"], errors="coerce"
                    )

                    uploaded_df["traffic_volume"] = pd.to_numeric(
                    uploaded_df["traffic_volume"], errors="coerce"
                    )

                    before_cleaning = len(uploaded_df)

                    uploaded_df = uploaded_df.dropna(
                        subset=["timestamp", "traffic_volume"]
                    )

                    uploaded_df = uploaded_df[
                        uploaded_df["traffic_volume"] >= 0
                    ]

                    road_column = (
                        "road_id" if "road_id" in uploaded_df.columns
                         else "road_name"
                    )

                    uploaded_df = uploaded_df.drop_duplicates(
                        subset=[road_column, "timestamp"]
                    )

                    removed_rows = before_cleaning - len(uploaded_df)

                    uploaded_df.to_csv(uploaded_path, index=False)

                    st.write("Rows removed during cleaning:", removed_rows)
                    st.write("Cleaned dataset shape:", uploaded_df.shape)

                    st.success("Validated traffic data saved successfully!")

            except Exception as error:
                st.error(f"Unable to read CSV: {error}")

    elif section == "Live Predictions":

        st.header("Live Traffic Predictions")
        st.info(
            "Predictions use the saved trained models and "
            "latest available processed road readings."
        )

        try:
            # 1. Load processed feature data
            prediction_df = load_feature_data()

            if prediction_df.empty:
                st.warning("No processed traffic data available.")

            else:
                # 2. Sort records by timestamp
                if "timestamp" in prediction_df.columns:
                    prediction_df["timestamp"] = pd.to_datetime(
                        prediction_df["timestamp"],
                        errors="coerce"
                    )
                    prediction_df = prediction_df.sort_values("timestamp")

                # 3. Identify road column
                if "road_name" in prediction_df.columns:
                    road_column = "road_name"
                elif "road_id" in prediction_df.columns:
                    road_column = "road_id"
                else:
                    raise ValueError(
                        "Road name or road ID column is missing."
                    )

                # 4. Get latest reading for each road
                latest = (
                    prediction_df
                    .groupby(road_column, as_index=False)
                    .tail(1)
                    .copy()
                )

                # 5. Traffic volume prediction
                rf_model = load_rf_model()
                rf_features = load_rf_features()

                if rf_model is not None and rf_features:
                    X_rf = latest.reindex(columns=rf_features)
                    X_rf = X_rf.apply(
                        pd.to_numeric,
                        errors="coerce"
                    ).fillna(0)

                    latest["Predicted Traffic Volume"] = (
                        rf_model.predict(X_rf)
                    )
                else:
                    latest["Predicted Traffic Volume"] = np.nan
                    st.warning(
                        "Traffic volume model or feature list is unavailable."
                    )
                    # 90% Traffic Volume Prediction Confidence Interval
                    error_margin = 115.28

                    latest["Lower Bound"] = (
                        latest["Predicted Traffic Volume"] - error_margin
                    ).clip(lower=0)

                    latest["Upper Bound"] = (
                        latest["Predicted Traffic Volume"] + error_margin
                    )

                # 6. Congestion prediction
                try:
                    congestion_model, congestion_features = (
                        load_congestion_model()
                    )

                    if (
                        congestion_model is not None
                        and congestion_features
                    ):
                        X_congestion = latest.reindex(
                            columns=congestion_features
                        )

                        X_congestion = X_congestion.apply(
                            pd.to_numeric,
                            errors="coerce"
                        ).fillna(0)

                        latest["Predicted Congestion"] = (
                            congestion_model.predict(X_congestion)
                        )
                    else:
                        latest["Predicted Congestion"] = "Unavailable"

                except Exception as error:
                    latest["Predicted Congestion"] = "Unavailable"
                    st.warning(
                        f"Congestion prediction unavailable: {error}"
                    )

                # 7. Accident risk prediction
                try:
                    (
                        accident_model,
                        accident_imputer,
                        accident_features,
                        accident_threshold
                    ) = load_accident_model()

                    X_accident = latest.reindex(
                        columns=accident_features
                    )

                    X_accident = X_accident.apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    X_accident_imputed = accident_imputer.transform(
                        X_accident
                    )

                    risk_probabilities = accident_model.predict_proba(
                        X_accident_imputed
                    )[:, 1]

                    latest["Accident Risk (%)"] = (
                        risk_probabilities * 100
                    ).round(2)

                    latest["Predicted Accident Risk"] = [
                        "High Risk" if p >= accident_threshold
                        else "Low Risk"
                        for p in risk_probabilities
                    ]

                except Exception as error:
                    latest["Accident Risk (%)"] = np.nan
                    latest["Predicted Accident Risk"] = "Unavailable"
                    st.warning(
                        f"Accident prediction unavailable: {error}"
                    )
                    
            # Ensure confidence interval columns always exist
            if "Predicted Traffic Volume" not in latest.columns:
                latest["Predicted Traffic Volume"] = np.nan

            error_margin = 115.28

            latest["Lower Bound"] = (
              latest["Predicted Traffic Volume"] - error_margin
            ).clip(lower=0)

            latest["Upper Bound"] = (
                latest["Predicted Traffic Volume"] + error_margin
            )
            # 8. Display results
            display_columns = [road_column]

            if "timestamp" in latest.columns:
                display_columns.append("timestamp")

            display_columns.extend([
                "Predicted Traffic Volume",
                "Lower Bound",
                "Upper Bound",
                "Predicted Congestion",
                "Accident Risk (%)",
                "Predicted Accident Risk"
             ])

            # Ensure all display columns exist
            required_columns = [
                "Predicted Traffic Volume",
                "Predicted Congestion",
                "Accident Risk (%)",
                "Predicted Accident Risk"
            ]

            for col in required_columns:
                if col not in latest.columns:
                    latest[col] = "Unavailable"

            st.subheader("Latest Predictions by Road")

            st.dataframe(
                latest[display_columns].round(2),
                use_container_width=True
            )

            st.caption(
                "Predictions are based on the latest available "
                "processed sensor readings; they are not real-time "
                "sensor feeds."
            )

        except Exception as error:
            st.error(f"Prediction error: {error}")

except Exception as error:
    st.error(f"Unable to load the FlowCast dashboard: {error}")