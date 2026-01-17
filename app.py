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

'''
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