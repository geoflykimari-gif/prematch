import pandas as pd
import unicodedata

# ---------------------------
# Championship Teams to Fix
# ---------------------------
CHAMP_TEAMS = {
    "portsmouth": "Portsmouth",
    "millwall": "Millwall",
    "oxfordunited": "Oxford United",
    "wrexham": "Wrexham",
}

# ---------------------------
# Normalize strings for matching
# ---------------------------
def normalize_str(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    # remove accents
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    # remove fc/afc, hyphens, spaces
    s = s.replace("fc", "").replace("afc", "").replace("-", "").replace(" ", "")
    return s

# ---------------------------
# Fix team name
# ---------------------------
def fix_team_name(name):
    norm = normalize_str(name)
    if norm in CHAMP_TEAMS:
        return CHAMP_TEAMS[norm]
    return name  # leave everything else unchanged

# ---------------------------
# Apply to CSV
# ---------------------------
def fix_championship_aliases(input_csv, output_csv):
    df = pd.read_csv(input_csv, dtype=str)

    if "home" in df.columns:
        df["home"] = df["home"].apply(fix_team_name)
    if "away" in df.columns:
        df["away"] = df["away"].apply(fix_team_name)

    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned CSV → {output_csv}")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    fix_championship_aliases("master_matches.csv", "master_matches_cleaned.csv")
