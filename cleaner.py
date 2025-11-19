import pandas as pd

# -------------------------------
# Load master CSV
# -------------------------------
df_master = pd.read_csv("master_matches.csv")

# -------------------------------
# Normalize team names
# -------------------------------
TEAM_ALIASES = {
    "Preston": "Preston North End",
    # Add other aliases as needed
}

def normalize_team(name):
    if not name: 
        return ""
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name.title())

df_master["home"] = df_master["home"].apply(normalize_team)
df_master["away"] = df_master["away"].apply(normalize_team)

# -------------------------------
# Save back the master CSV
# -------------------------------
df_master.to_csv("master_matches.csv", index=False)
print("Master CSV updated with normalized team names")
