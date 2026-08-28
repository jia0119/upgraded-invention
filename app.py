import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Seoul Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("SeoulBikeData.csv")
    # Clean column names to remove special characters/spaces for easier access
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# -----------------------------------------------------------------------------
# 3. Sidebar Filters
# -----------------------------------------------------------------------------
st.sidebar.header("Filter Options")

# Season Filter
seasons = df["Seasons"].unique().tolist()
selected_seasons = st.sidebar.multiselect(
    "Select Season(s):",
    options=seasons,
    default=seasons
)

# Holiday Filter
holidays = df["Holiday"].unique().tolist()
selected_holidays = st.sidebar.multiselect(
    "Select Holiday Status:",
    options=holidays,
    default=holidays
)

# Temperature Range Filter
min_temp = float(df["Temperature(°C)"].min())
max_temp = float(df["Temperature(°C)"].max())
temp_range = st.sidebar.slider(
    "Temperature Range (°C):",
    min_value=min_temp,
    max_value=max_temp,
    value=(min_temp, max_temp)
)

# Apply Filters to Dataframe
filtered_df = df[
    (df["Seasons"].isin(selected_seasons)) &
    (df["Holiday"].isin(selected_holidays)) &
    (df["Temperature(°C)"].between(temp_range[0], temp_range[1]))
]

# -----------------------------------------------------------------------------
# 4. Dashboard Header & Key Metrics (KPIs)
# -----------------------------------------------------------------------------
st.title("🚲 Seoul Bike Sharing Demand Dashboard")
st.markdown("Explore key metrics, distributions, and trends based on environmental factors.")

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_rentals = int(filtered_df["Rented Bike Count"].sum())
    st.metric(label="Total Rented Bikes", value=f"{total_rentals:,}")

with col2:
    avg_rentals = round(filtered_df["Rented Bike Count"].mean(), 1)
    st.metric(label="Avg Hourly Rentals", value=f"{avg_rentals:,}")

with col3:
    avg_temp = round(filtered_df["Temperature(°C)"].mean(), 1)
    st.metric(label="Avg Temperature (°C)", value=f"{avg_temp}°C")

with col4:
    total_records = len(filtered_df)
    st.metric(label="Total Recorded Hours", value=f"{total_records:,}")

st.divider()

# -----------------------------------------------------------------------------
# 5. Data Visualizations Section
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Distribution Analysis", "⏰ Hourly & Seasonal Trends", "📄 Raw Data View"])

# TAB 1: Distribution Analysis
with tab1:
    st.subheader("Distribution of Rented Bike Count")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        filtered_df["Rented Bike Count"],
        bins=30,
        color="skyblue",
        edgecolor="black"
    )
    ax.set_title("Distribution of Rented Bike Count")
    ax.set_xlabel("Rented Bike Count")
    ax.set_ylabel("Frequency")
    
    st.pyplot(fig)

# TAB 2: Hourly & Seasonal Trends
with tab2:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Average Bike Rentals by Hour")
        hourly_avg = filtered_df.groupby("Hour")["Rented Bike Count"].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=hourly_avg, x="Hour", y="Rented Bike Count", marker="o", ax=ax, color="#2b5c8f")
        ax.set_ylabel("Average Rented Bikes")
        ax.set_xlabel("Hour of Day")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
        
    with col_right:
        st.subheader("Average Bike Rentals by Season")
        season_avg = filtered_df.groupby("Seasons")["Rented Bike Count"].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=season_avg, x="Seasons", y="Rented Bike Count", palette="viridis", ax=ax)
        ax.set_ylabel("Average Rented Bikes")
        st.pyplot(fig)

# TAB 3: Raw Data & Summary Statistics
with tab3:
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df, use_container_width=True)
    
    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe(), use_container_width=True)