import pandas as pd

# --- 1. LOAD THE DATASETS ---
try:
    df_enrol = pd.read_csv('enrolment_full1.csv')
    df_demo = pd.read_csv('merged_demographic.csv')
    df_bio = pd.read_csv('merged_biometric.csv')
    
    print("✅ All 3 files loaded successfully!")
    
except FileNotFoundError as e:
    print(f"❌ Error: {e}")

# --- 2. PRINT COLUMN NAMES ---
# We need to see these to write the Merge code in the next step
print("\n--- Enrollment Columns ---")
print(list(df_enrol.columns))

print("\n--- Demographic Update Columns ---")
print(list(df_demo.columns))

print("\n--- Biometric Update Columns ---")
print(list(df_bio.columns))

# --- 3. BASIC PEEK ---
print("\n--- First 2 rows of Enrollment ---")
print(df_enrol.head(2))

# --- STEP 2: CLEAN, MERGE & CALCULATE ---

# 1. Fix Dates (Handles the "DtypeWarning")
# usage of errors='coerce' turns bad data into NaT (Not a Time) so it doesn't crash
df_enrol['date'] = pd.to_datetime(df_enrol['date'], dayfirst=True, errors='coerce')
df_demo['date'] = pd.to_datetime(df_demo['date'], dayfirst=True, errors='coerce')
df_bio['date'] = pd.to_datetime(df_bio['date'], dayfirst=True, errors='coerce')

# Drop rows with invalid dates
df_enrol.dropna(subset=['date'], inplace=True)
df_demo.dropna(subset=['date'], inplace=True)
df_bio.dropna(subset=['date'], inplace=True)

# 2. Aggregate by Date, State, and District
# We group by these 3 columns and SUM the numbers. 
# This aligns the data so we can merge (e.g., matching "Pune" in file 1 with "Pune" in file 2)
group_cols = ['date', 'state', 'district']

enrol_grouped = df_enrol.groupby(group_cols)[['age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()
demo_grouped = df_demo.groupby(group_cols)[['demo_age_5_17', 'demo_age_17_']].sum().reset_index()
bio_grouped = df_bio.groupby(group_cols)[['bio_age_5_17', 'bio_age_17_']].sum().reset_index()

# 3. Merge into one MASTER DataFrame
# We use 'outer' merge to keep all data (even if a district has updates but no new enrolments that day)
master_df = pd.merge(enrol_grouped, demo_grouped, on=group_cols, how='outer')
master_df = pd.merge(master_df, bio_grouped, on=group_cols, how='outer')

# Fill missing values with 0 (NaN means 0 transactions happened)
master_df = master_df.fillna(0)

# 4. Calculate Totals & AUSI (Stress Index)
# Total Enrolments
master_df['Total_Enrolments'] = (master_df['age_0_5'] + 
                                 master_df['age_5_17'] + 
                                 master_df['age_18_greater'])

# Total Updates (Demographic + Biometric)
master_df['Total_Updates'] = (master_df['demo_age_5_17'] + master_df['demo_age_17_'] + 
                              master_df['bio_age_5_17'] + master_df['bio_age_17_'])

# Calculate AUSI: Updates / Enrolments
# We add +1 to the denominator to avoid dividing by zero error
master_df['AUSI'] = master_df['Total_Updates'] / (master_df['Total_Enrolments'] + 1)

# --- VERIFY ---
print("✅ Master DataFrame Created!")
print(master_df[['date', 'district', 'Total_Enrolments', 'Total_Updates', 'AUSI']].head())
print(f"\nTotal rows in Master Data: {len(master_df)}")

import matplotlib.pyplot as plt

# 1. Quick Sanity Check Plot
# We aggregate everything by date to see the "National Trend"
daily_trends = master_df.groupby('date')[['Total_Enrolments', 'Total_Updates']].sum()

plt.figure(figsize=(10, 5))
plt.plot(daily_trends.index, daily_trends['Total_Enrolments'], label='Enrolments (New)', color='green')
plt.plot(daily_trends.index, daily_trends['Total_Updates'], label='Updates (Changes)', color='orange')
plt.title('Sanity Check: Daily Trends (National)')
plt.legend()
plt.grid(True)
plt.show()

# 2. SAVE THE FILE (Crucial for the App)
master_df.to_csv('aadhaar_master.csv', index=False)
print("✅ Data saved as 'aadhaar_master.csv'. You are ready to launch the App!")
