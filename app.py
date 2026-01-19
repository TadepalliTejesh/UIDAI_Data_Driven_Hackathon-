'''
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------- PAGE CONFIG (ONLY ONCE) ----------
st.set_page_config(
    page_title="Aadhaar Dashboard",
    layout="wide"
)
# ---------- LOAD DATA ----------
df = pd.read_csv("enrolment_full1.csv")
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center;'>📊 Aadhaar Enrolment & Update Analysis (2025)</h1>
    <p style='text-align: center; color: gray;'>
    Unlocking societal trends to support UIDAI decision-making
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------- SIDEBAR FILTER ----------
st.sidebar.header("🔍 Filter by State")
selected_states = st.sidebar.multiselect(
    "Select States",
    options=sorted(df['state'].unique()),
    default=sorted(df['state'].unique())
)

df = df[df['state'].isin(selected_states)]

# ======================================================
# 📌 SECTION 1: STATE-WISE ENROLMENT PATTERNS
# ======================================================

st.subheader("📍 State-wise Aadhaar Enrolment by Age Group")

state_totals = (
    df.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater']]
    .sum()
    .sort_values(by='age_18_greater', ascending=False)
)

# ---------- METRICS ----------
col1, col2, col3 = st.columns(3)

col1.metric("👶 Age 0-5", f"{int(state_totals['age_0_5'].sum()):,}")
col2.metric("🧒 Age 5-17", f"{int(state_totals['age_5_17'].sum()):,}")
col3.metric("🧑 Age 18+", f"{int(state_totals['age_18_greater'].sum()):,}")

# ---------- BAR CHART ----------
fig1, ax1 = plt.subplots(figsize=(14,6))
state_totals.plot(kind='bar', ax=ax1)

ax1.set_xlabel("State")
ax1.set_ylabel("Total Enrolments")
ax1.set_title("State-wise Aadhaar Enrolment Distribution")
plt.xticks(rotation=45, ha="right")

st.pyplot(fig1)

# ---------- INSIGHTS ----------
st.markdown("""
### 🧠 Key Insights
- **18+ age group dominates enrolment** across most states  
- Child enrolment (0–5) varies significantly by region  
- Indicates delayed enrolment and uneven early-age coverage  

### 🛠 Suggested Action
- Strengthen Aadhaar enrolment at birth & school entry  
- Deploy mobile enrolment units in low-performing states  
""")

# ======================================================
# 📈 SECTION 2: MONTHLY TRENDS & PREDICTIVE INDICATORS
# ======================================================

st.subheader("📈 Monthly Aadhaar Enrolment Trends by Age Group")

monthly = (
    df
    .groupby(df['date'].dt.to_period('M'))
    [['age_0_5', 'age_5_17', 'age_18_greater']]
    .sum()
)

monthly.index = monthly.index.to_timestamp()

# ---------- MOVING AVERAGE ----------
monthly_ma = monthly.rolling(window=3).mean()

# ---------- LINE CHART ----------
fig2, ax2 = plt.subplots(figsize=(12,5))

monthly.plot(ax=ax2)
monthly_ma.plot(ax=ax2, linestyle='--')

ax2.set_xlabel("Month")
ax2.set_ylabel("Number of Enrolments")
ax2.set_title("Monthly Aadhaar Enrolment Trends (with 3-Month Moving Average)")

st.pyplot(fig2)

# ---------- SEASONAL PEAK ----------
peak_month = monthly['age_0_5'].idxmax()
st.info(f"📌 Peak child (0-5) enrolment observed in **{peak_month.strftime('%B %Y')}**")

# ---------- INSIGHTS ----------
st.markdown("""
### 🧠 Trend Insights
- Adult enrolment remains consistently dominant  
- Child enrolment shows **seasonal peaks**  
- Overall stability enables **predictive planning**

### 🛠 Policy Implications
- Scale enrolment centers during peak months  
- Use trend stability for workforce planning  
- Adopt data-driven forecasting for UIDAI operations  
""")

# ======================================================
# 📌 SECTION 3: DISTRICT-WISE ENROLMENT PATTERNS
# ======================================================

district_enrolment = (
    df
    .groupby(['state', 'district'])
    [['age_0_5', 'age_5_17', 'age_18_greater']]
    .sum()
    .sort_values(by='age_18_greater', ascending=False)
)

# ---------- TOP 10 DISTRICTS ----------
top_districts = district_enrolment.head(10)

# ---------- BAR CHART ----------
fig, ax = plt.subplots(figsize=(12,5))

top_districts[['age_0_5', 'age_5_17', 'age_18_greater']].plot(
    kind='bar',
    ax=ax
)

ax.set_title("Top 10 Districts by Aadhaar Enrolment (2025)")
ax.set_xlabel("State - District")
ax.set_ylabel("Total Enrolments")
plt.xticks(rotation=45, ha="right")

st.pyplot(fig)

# ---------- INSIGHTS ----------
st.markdown("""
### 🧠 Key Insights
- Aadhaar enrolment is **highly concentrated in a few districts**
- Adult (18+) enrolment dominates across top districts
- Indicates **uneven regional workload**

### 🛠 Policy Implications
- Deploy additional enrolment infrastructure in high-demand districts
- Use district-level planning instead of only state-level
- Reduce service delays by proactive resource allocation
""")

# ===============================================
# 📌 SECTION 4: DISTRICT-LEVEL ANOMALY DETECTION 
# ===============================================

st.subheader("🚨 District-level Anomaly Detection")

# ---------- AGGREGATE DISTRICT DATA ----------
district_totals = (
    df.groupby(['state', 'district'])[['age_0_5', 'age_5_17', 'age_18_greater']]
    .sum()
    .reset_index()
)

# ---------- TOTAL ENROLMENT ----------
district_totals['total_enrolment'] = (
    district_totals['age_0_5'] +
    district_totals['age_5_17'] +
    district_totals['age_18_greater']
)

# ---------- Z-SCORE CALCULATION ----------
mean_val = district_totals['total_enrolment'].mean()
std_val = district_totals['total_enrolment'].std()

district_totals['z_score'] = (
    (district_totals['total_enrolment'] - mean_val) / std_val
)

# ---------- FLAG ANOMALIES ----------
anomalies = district_totals[
    (district_totals['z_score'] > 2) | (district_totals['z_score'] < -2)
]

# ---------- DISPLAY ----------
st.write("📌 Districts with unusually high or low enrolment:")
st.dataframe(
    anomalies[['state', 'district', 'total_enrolment', 'z_score']]
    .sort_values(by='z_score', ascending=False)
)

fig, ax = plt.subplots(figsize=(12,5))

ax.scatter(
    district_totals.index,
    district_totals['total_enrolment'],
    alpha=0.5
)

ax.scatter(
    anomalies.index,
    anomalies['total_enrolment'],
    color='red',
    label='Anomalies'
)

ax.set_title("District Enrolment Anomaly Detection")
ax.set_xlabel("District Index")
ax.set_ylabel("Total Enrolments")
ax.legend()

st.pyplot(fig)

# ---------- FOOTER ----------
st.markdown(
    "<hr><p style='text-align:center; color:gray;'>UIDAI Data • Hackathon Presentation</p>",
    unsafe_allow_html=True
)


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import os
from sklearn.linear_model import LinearRegression

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Aadhaar Intelligence Dashboard", layout="wide")

# ---------- LOAD ENROLMENT DATA ----------
df = pd.read_csv("enrolment_full1.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# ---------- LOAD DEMOGRAPHIC ZIP ----------
with zipfile.ZipFile("api_data_aadhar_demographic.zip", 'r') as zip_ref:
    zip_ref.extractall("demo_data")

demo_folder = os.listdir("demo_data")[0]
demo_csv = os.listdir("demo_data/" + demo_folder)[0]
df_demo = pd.read_csv("demo_data/" + demo_folder + "/" + demo_csv)

# FIX: Create proper datetime column in df_demo
df_demo['date'] = pd.to_datetime(df_demo.iloc[:,0], errors='coerce')
df_demo = df_demo.dropna(subset=['date'])

# ---------- LOAD BIOMETRIC ZIP ----------
with zipfile.ZipFile("api_data_aadhar_biometric.zip", 'r') as zip_ref:
    zip_ref.extractall("bio_data")

bio_folder = os.listdir("bio_data")[0]
bio_csv = os.listdir("bio_data/" + bio_folder)[0]
df_bio = pd.read_csv("bio_data/" + bio_folder + "/" + bio_csv)

# FIX: Create proper datetime column in df_bio
df_bio['date'] = pd.to_datetime(df_bio.iloc[:,0], errors='coerce')
df_bio = df_bio.dropna(subset=['date'])

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center;'>📊 Aadhaar Service Intelligence Platform</h1>
<p style='text-align:center; color:gray;'>UIDAI Hackathon Project</p>
<hr>
""", unsafe_allow_html=True)

# ---------- SIDEBAR STATE FILTER ----------
st.sidebar.header("🔍 Filter by State")
selected_states = st.sidebar.multiselect(
    "Select States",
    options=sorted(df['state'].unique()),
    default=sorted(df['state'].unique())
)
df = df[df['state'].isin(selected_states)]

# ======================================================
# SECTION 1: STATE-WISE ENROLMENT  (Friend's Code)
# ======================================================

st.subheader("📍 State-wise Aadhaar Enrolment by Age Group")

state_totals = df.groupby('state')[['age_0_5','age_5_17','age_18_greater']].sum()

fig1, ax1 = plt.subplots(figsize=(14,6))
state_totals.plot(kind='bar', ax=ax1)
ax1.set_title("State-wise Enrolment Distribution")
st.pyplot(fig1)

# ======================================================
# SECTION 2: MONTHLY TREND (Friend's Code)
# ======================================================

st.subheader("📈 Monthly Enrolment Trends")

monthly = df.groupby(df['date'].dt.to_period('M'))[['age_0_5','age_5_17','age_18_greater']].sum()
monthly.index = monthly.index.to_timestamp()

fig2, ax2 = plt.subplots(figsize=(12,5))
monthly.plot(ax=ax2)
ax2.set_title("Monthly Aadhaar Enrolment Trends")
st.pyplot(fig2)

# ======================================================
# SECTION 3: ANOMALY DETECTION (Friend's Code)
# ======================================================

st.subheader("🚨 District-level Anomaly Detection")

district_totals = df.groupby(['state','district'])[['age_0_5','age_5_17','age_18_greater']].sum()
district_totals['total'] = district_totals.sum(axis=1)

mean_val = district_totals['total'].mean()
std_val = district_totals['total'].std()
district_totals['z'] = (district_totals['total'] - mean_val) / std_val

anomalies = district_totals[(district_totals['z']>2)|(district_totals['z']<-2)]
st.dataframe(anomalies.sort_values(by='z', ascending=False))

# ======================================================
# SECTION 4: STRESS INDEX (AUSI)
# ======================================================

st.subheader("🔥 Aadhaar Update Stress Index (AUSI)")

# Monthly enrolments
enrol_monthly = df.groupby(df['date'].dt.to_period('M'))[['age_0_5','age_5_17','age_18_greater']].sum()
enrol_monthly['new_enrolments'] = enrol_monthly.sum(axis=1)

# Monthly demographic updates
demo_monthly = df_demo.groupby(df_demo['date'].dt.to_period('M')).sum(numeric_only=True)
demo_monthly['demo_updates'] = demo_monthly.sum(axis=1)

# Monthly biometric updates
bio_monthly = df_bio.groupby(df_bio['date'].dt.to_period('M')).sum(numeric_only=True)
bio_monthly['bio_updates'] = bio_monthly.sum(axis=1)

# Combine safely
stress = pd.concat(
    [enrol_monthly['new_enrolments'],
     demo_monthly['demo_updates'],
     bio_monthly['bio_updates']], axis=1)

stress.columns = ['new_enrolments','demo_updates','bio_updates']
stress = stress.dropna()

stress['AUSI'] = (stress['demo_updates'] + stress['bio_updates']) / stress['new_enrolments']
stress.index = stress.index.to_timestamp()

fig3, ax3 = plt.subplots(figsize=(12,5))
stress['AUSI'].plot(ax=ax3, color='red')
ax3.set_title("Aadhaar Update Stress Index Over Time")
st.pyplot(fig3)

latest_ausi = stress['AUSI'].iloc[-1]

if latest_ausi > 2:
    st.warning("⚠️ High Update Stress → Recommend deploying additional update centers")
else:
    st.success("✅ Update Load Normal")

# ======================================================
# SECTION 5: FORECASTING
# ======================================================

st.subheader("🔮 Next Month Enrolment Forecast")

y = enrol_monthly['new_enrolments'].values
X = np.arange(len(y)).reshape(-1,1)

model = LinearRegression()
model.fit(X,y)

next_month = model.predict([[len(y)]])[0]
st.metric("Predicted Next Month Enrolments", f"{int(next_month):,}")

# ======================================================
# SECTION 6: BIOMETRIC REMINDER SYSTEM
# ======================================================

st.subheader("👶 Biometric Update Reminder System")

st.info("""
Children turning **5 or 15 years** must update Aadhaar biometrics.

📩 Recommendation:
Send reminder SMS to parents / guardians of children approaching biometric update age.

This reduces last-minute rush at UIDAI centers.
""")

# ======================================================
# SECTION 7: POLICY RECOMMENDATIONS
# ======================================================

st.subheader("🏛️ Automated Policy Recommendations")

if latest_ausi > 2:
    st.error("""
**Recommendations**
- Deploy mobile biometric update vans  
- Increase staffing at high-stress districts  
- Extend working hours temporarily
""")
else:
    st.success("""
**Recommendations**
- Current infrastructure sufficient  
- Continue routine monitoring
""")

# ---------- FOOTER ----------
st.markdown("<hr><p style='text-align:center;color:gray;'>UIDAI Hackathon Project • Team Submission</p>", unsafe_allow_html=True)
'''

import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="ASIP Dashboard", layout="wide", page_icon="🇮🇳")

st.title("🇮🇳 Aadhaar Service Intelligence Platform (ASIP)")
st.markdown("### Operational Intelligence for UIDAI Enrolment & Update Centers")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. LOAD DATA
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Load the Master Dataset created in Step 3
        df = pd.read_csv('aadhaar_master.csv')
        # .dt.date removes the time component (00:00:00)
        df['date'] = pd.to_datetime(df['date']).dt.date  
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("❌ Critical Error: 'aadhaar_master.csv' not found. Please run the Data Processing Notebook first!")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Filter Data")

# State Filter
state_list = ["All"] + sorted(list(df['state'].unique()))
selected_state = st.sidebar.selectbox("Select State", state_list)

if selected_state != "All":
    df_filtered = df[df['state'] == selected_state]
    # District Filter (Only shows districts in selected state)
    district_list = ["All"] + sorted(list(df_filtered['district'].unique()))
    selected_district = st.sidebar.selectbox("Select District", district_list)
    
    if selected_district != "All":
        df_filtered = df_filtered[df_filtered['district'] == selected_district]
else:
    df_filtered = df

st.sidebar.markdown("---")
st.sidebar.info(f"**Data Loaded:** {len(df_filtered):,} rows")

# -----------------------------------------------------------------------------
# 4. KPI METRICS (Top Row)
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_enrol = df_filtered['Total_Enrolments'].sum()
total_updates = df_filtered['Total_Updates'].sum()
avg_ausi = df_filtered['AUSI'].mean()
# Calculate % Biometric (Insightful Metric)
bio_updates = df_filtered['bio_age_5_17'].sum() + df_filtered['bio_age_17_'].sum()
bio_pct = (bio_updates / total_updates * 100) if total_updates > 0 else 0

col1.metric("Total Enrolments (New)", f"{total_enrol:,.0f}")
col2.metric("Total Updates (Changes)", f"{total_updates:,.0f}")
col3.metric("Avg Stress Index (AUSI)", f"{avg_ausi:.2f}", help="Updates divided by Enrolments. Higher = More Pressure.")
col4.metric("Biometric Share", f"{bio_pct:.1f}%", help="Percentage of updates that are Biometric")

# -----------------------------------------------------------------------------
# 5. MAIN ANALYSIS TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trend Explorer", "🚨 Stress Index (AUSI)", "⚠️ Anomaly Detector", "📢 Action Center", "🔮 Demand Forecast"])

# --- TAB 1: TRENDS (The Split View) ---
with tab1:
    st.subheader("Daily Transaction Trends (Split View)")
    st.write("Breakdown of Workload: New Enrolments vs. Demographic Updates vs. Biometric Updates.")
    
    # Aggregate by date
    trend_data = df_filtered.groupby('date').agg({
        'Total_Enrolments': 'sum',
        'demo_age_5_17': 'sum',
        'demo_age_17_': 'sum',
        'bio_age_5_17': 'sum',
        'bio_age_17_': 'sum'
    }).reset_index()

    # Create explicit columns for the graph
    trend_data['Demographic Updates'] = trend_data['demo_age_5_17'] + trend_data['demo_age_17_']
    trend_data['Biometric Updates'] = trend_data['bio_age_5_17'] + trend_data['bio_age_17_']

    # Plot
    fig_trend = px.line(trend_data, x='date', 
                        y=['Total_Enrolments', 'Demographic Updates', 'Biometric Updates'],
                        title="Workload Volume Over Time",
                        color_discrete_map={
                            "Total_Enrolments": "green", 
                            "Demographic Updates": "blue", 
                            "Biometric Updates": "orange"
                        },
                        markers=True)
    
    st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 2: AUSI (Stress Index) ---
with tab2:
    st.subheader("📍 District Stress Heatmap")
    st.write("Which districts are spending more time on Updates than new Enrolments?")
    
    # Group by District
    district_stats = df_filtered.groupby('district').agg({
        'Total_Enrolments': 'sum',
        'Total_Updates': 'sum',
        'AUSI': 'mean' # Average daily stress
    }).reset_index()
    
    # Sort by Stress
    top_stressed = district_stats.sort_values(by='AUSI', ascending=False).head(15)
    
    fig_bar = px.bar(top_stressed, x='district', y='AUSI',
                     color='AUSI', title="Top 15 High-Stress Districts",
                     color_continuous_scale='Reds',
                     text_auto='.1f')
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.info("💡 **Insight:** High AUSI (> 5.0) indicates the center is overwhelmed by maintenance/update work rather than onboarding new citizens.")

# --- REPLACEMENT FOR TAB 3 (ANOMALY DETECTOR) ---
with tab3:
    st.subheader("🔍 Operational Anomalies")
    st.write("Detecting unusual spikes in activity (Potential Fraud or Migration Events).")
    
    # 1. Calculate Statistics
    avg_updates = df_filtered['Total_Updates'].mean()
    threshold = df_filtered['Total_Updates'].quantile(0.95)
    
    # 2. Filter Anomalies
    anomalies = df_filtered[df_filtered['Total_Updates'] > threshold].copy()
    anomalies = anomalies.sort_values(by='Total_Updates', ascending=False)
    
    # 3. ADD THE "WHY" (Reasoning Column)
    # This calculates how many times higher than average the volume is (e.g., "3.5x Normal")
    anomalies['Severity'] = anomalies['Total_Updates'] / avg_updates
    anomalies['Reason'] = anomalies['Severity'].apply(lambda x: f"Volume is {x:.1f}x higher than average")

    # Metrics
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Typical Daily Volume", f"{avg_updates:,.0f}")
    col_m2.metric("Anomaly Threshold (95%)", f"{threshold:,.0f}")
    
    if not anomalies.empty:
        st.warning(f"⚠️ Found {len(anomalies)} suspicious high-activity days.")
        # Show the new 'Reason' column so the user understands WHY it's an anomaly
        st.dataframe(anomalies[['date', 'district', 'Total_Updates', 'Reason']].head(15), use_container_width=True)
    else:
        st.success("✅ No anomalies detected. Operations are stable.")

# --- TAB 4: ACTION CENTER (RECOMMENDATIONS) ---
with tab4:
    st.subheader("🚀 Recommended Actions")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 📡 Mobile Unit Deployment")
        st.write("Districts with **High Stress (AUSI > 10)** need mobile units to clear the update backlog.")
        
        # Logic: Find districts with high AUSI
        high_stress = district_stats[district_stats['AUSI'] > 10]
        
        if not high_stress.empty:
            for dist in high_stress['district'].head(5):
                st.error(f"🚨 **Deploy Unit to:** {dist} (AUSI: {high_stress[high_stress['district']==dist]['AUSI'].values[0]:.1f})")
        else:
            st.success("No districts currently require emergency deployment.")

    with col_b:
        st.markdown("### 🔔 Biometric Update Reminders")
        st.write("Based on Age 5-17 data, these districts need 'Mandatory Biometric Update' camps.")
        
        # Logic: Districts with high volume of bio updates pending (using proxy of high bio activity)
        # We look for places with High Bio Updates
        high_bio = district_stats.sort_values(by='Total_Updates', ascending=False).head(5)
        
        for dist in high_bio['district']:
            st.info(f"📩 **Send SMS Blast in:** {dist}")

    st.markdown("---")
    st.caption("Generated by ASIP Intelligence Engine")

# --- TAB 5: DEMAND FORECAST (The Missing Piece) ---
with tab5:
    st.subheader("🔮 Workload Forecast (Next 7 Days)")
    st.write("Predicting upcoming volume to help with staff scheduling.")
    
    # 1. Prepare Data
    # We use the last 30 days to calculate a rolling average
    daily_counts = df_filtered.groupby('date')[['Total_Updates']].sum().reset_index()
    daily_counts = daily_counts.sort_values('date')
    
    # Calculate 7-Day Moving Average
    daily_counts['Moving_Avg'] = daily_counts['Total_Updates'].rolling(window=7).mean()
    
    # 2. "Predict" the next 7 days
    last_avg_val = daily_counts['Moving_Avg'].iloc[-1]
    last_date = daily_counts['date'].max()
    
    # Create future dates
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'Predicted_Updates': [last_avg_val] * 7 # Simple baseline forecast
    })
    
    # 3. Visuals
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.metric("Predicted Daily Load", f"{last_avg_val:,.0f} updates/day")
    with col_f2:
        st.info("Based on 7-Day Moving Average Trend")
        
    # Plot Historical vs Forecast
    fig_forecast = px.line(daily_counts.tail(30), x='date', y='Total_Updates', title="Historical (Last 30 Days) vs Forecast")
    fig_forecast.add_scatter(x=forecast_df['date'], y=forecast_df['Predicted_Updates'], mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash'))
    
    st.plotly_chart(fig_forecast, use_container_width=True)

# -----------------------------------------------------------------------------