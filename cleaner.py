import pandas as pd
import os

# -----------------------------
# File paths (default CSVs)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(BASE_DIR, "master_matches.csv")
UPCOMING_CSV = os.path.join(BASE_DIR, "upcoming_matches.csv")
MASTER_OUT = os.path.join(BASE_DIR, "master_cleaned.csv")
UPCOMING_OUT = os.path.join(BASE_DIR, "upcoming_cleaned.csv")

# -----------------------------
# Load CSVs
# -----------------------------
df_master = pd.read_csv(MASTER_CSV)
df_upcoming = pd.read_csv(UPCOMING_CSV)

# Strip column names
df_master.columns = [c.strip() for c in df_master.columns]
df_upcoming.columns = [c.strip() for c in df_upcoming.columns]

# -----------------------------
# Detect unique team names
# -----------------------------
master_teams = set(df_master['home'].dropna().unique()) | set(df_master['away'].dropna().unique())
upcoming_teams = set(df_upcoming['home'].dropna().unique()) | set(df_upcoming['away'].dropna().unique())

all_teams = master_teams | upcoming_teams

# -----------------------------
# Predefined strong aliases
# -----------------------------
TEAM_ALIASES = {
    # EPL
    "Man City": "Manchester City",
    "ManCity": "Manchester City",
    "Manchester City FC": "Manchester City",
    "Man Utd": "Manchester United",
    "Man United": "Manchester United",
    "Liverpool FC": "Liverpool",
    "Chelsea FC": "Chelsea",
    "Arsenal FC": "Arsenal",
    "Brighton & Hove Albion FC": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Brighton FC": "Brighton",
    # Championship
    "Bristol City Fc": "Bristol City",
    "Birmingham City Fc": "Birmingham City",
    "Leicester City FC": "Leicester City",
    "Oxford United FC": "Oxford United",
    "Millwall FC": "Millwall",
    "Middlesbrough FC": "Middlesbrough",
    "Preston North End FC": "Preston North End",
    "Coventry City FC": "Coventry City",
    "Norwich City FC": "Norwich City",
    "Sheffield Wednesday FC": "Sheffield Wednesday",
    "Derby County FC": "Derby County",
    "West Bromwich Albion FC": "West Bromwich Albion",
    "Hull City AFC": "Hull City",
    "Swansea City AFC": "Swansea City",
    "Portsmouth FC": "Portsmouth",
    "Charlton Athletic FC": "Charlton Athletic",
    # BelgianProLeague examples
    "Standard Liège": "Standard Liege",
    "RSC Anderlecht": "Anderlecht",
    # Add all remaining known aliases here
}

# -----------------------------
# Automatically detect unmatched teams
# -----------------------------
unmatched = [t for t in all_teams if t not in TEAM_ALIASES.values() and t not in TEAM_ALIASES.keys()]
if unmatched:
    print("Unmatched teams detected. Consider adding to TEAM_ALIASES:", unmatched)

# -----------------------------
# Normalize team names function
# -----------------------------
def normalize_team(name):
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)

# -----------------------------
# Apply normalization
# -----------------------------
for df in [df_master, df_upcoming]:
    for col in ['home', 'away']:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team)

# -----------------------------
# Save cleaned CSVs (order preserved)
# -----------------------------
df_master.to_csv(MASTER_OUT, index=False)
df_upcoming.to_csv(UPCOMING_OUT, index=False)

print(f"Master CSV cleaned -> {MASTER_OUT}")
print(f"Upcoming CSV cleaned -> {UPCOMING_OUT}")
print("All team names normalized robustly. Order, dates, and times preserved.")
