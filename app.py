import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime
import os

app = Flask(__name__)

# -----------------------------
# File paths (Render-friendly)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(BASE_DIR, "master_matches_cleaned.csv")
UPCOMING_CSV = os.path.join(BASE_DIR, "upcoming_cleaned.csv")

df_master = None
df_upcoming = None

# -----------------------------
# League dominance order
# -----------------------------
LEAGUE_ORDER = [
    "EPL",
    "Championship",
    "LaLiga",
    "SerieA",
    "Bundesliga",
    "Ligue1",
    "BelgianProLeague",
    "Eredivisie",
    "PrimeiraLiga"
]

# -----------------------------
# Team normalization (expand aliases)
# -----------------------------
TEAM_ALIASES = {
    # EPL
    "Man City": "Manchester City",
    "ManCity": "Manchester City",
    "Manchester City FC": "Manchester City",
    "Man Utd": "Manchester United",
    "Man United": "Manchester United",
    "Chelsea FC": "Chelsea",
    "Chelsea": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Liverpool": "Liverpool",
    "Arsenal FC": "Arsenal",
    "Arsenal": "Arsenal",
    "Brighton & Hove Albion": "Brighton",
    # Championship (minimal robust aliases)
    "PortsmouthFC": "Portsmouth",
    "Millwall FC": "Millwall",
    "Oxford United FC": "Oxford United",
    "Wrexham AFC": "Wrexham",
    # Bundesliga example
    "Sport Club Freiburg": "SC Freiburg",
}

def normalize_team(name):
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)

# -----------------------------
# Load master CSV
# -----------------------------
def load_master():
    global df_master
    try:
        df_master = pd.read_csv(MASTER_CSV)
    except FileNotFoundError:
        raise FileNotFoundError(f"Master CSV not found: {MASTER_CSV}")

    # Clean column names
    df_master.columns = [c.strip() for c in df_master.columns]

    # Fill missing goal values
    for col in ["fthg", "ftag"]:
        if col in df_master.columns:
            df_master[col] = pd.to_numeric(df_master[col], errors="coerce").fillna(0).astype(int)
    
    # Rename for consistency
    if "fthg" in df_master.columns: df_master.rename(columns={"fthg":"home_goals"}, inplace=True)
    if "ftag" in df_master.columns: df_master.rename(columns={"ftag":"away_goals"}, inplace=True)

    # Normalize team names
    df_master["home"] = df_master["home"].apply(normalize_team)
    df_master["away"] = df_master["away"].apply(normalize_team)

    # Combine date and optional time
    if "time" not in df_master.columns: df_master["time"] = "00:00"
    df_master["time"] = df_master["time"].fillna("00:00")
    df_master["datetime"] = pd.to_datetime(df_master["date"].astype(str) + " " + df_master["time"], errors="coerce")

# -----------------------------
# Load upcoming CSV
# -----------------------------
def load_upcoming():
    global df_upcoming
    try:
        df_upcoming = pd.read_csv(UPCOMING_CSV)
    except FileNotFoundError:
        raise FileNotFoundError(f"Upcoming CSV not found: {UPCOMING_CSV}")

    df_upcoming.columns = [c.strip() for c in df_upcoming.columns]
    df_upcoming["datetime"] = pd.to_datetime(df_upcoming["datetime"], errors="coerce")
    df_upcoming["date"] = df_upcoming["datetime"].dt.date.astype(str)
    df_upcoming["time"] = df_upcoming["datetime"].dt.strftime("%H:%M")
    df_upcoming["home"] = df_upcoming["home"].apply(normalize_team)
    df_upcoming["away"] = df_upcoming["away"].apply(normalize_team)

# -----------------------------
# Last N matches function
# -----------------------------
def get_last_n_matches(team, n=10):
    if df_master is None or df_master.empty:
        return []

    matches = df_master[
        (df_master["home"] == team) | (df_master["away"] == team)
    ].sort_values("datetime", ascending=False).head(n)

    results = []
    for _, row in matches.iterrows():
        if row["home"] == team:
            gf = row["home_goals"]
            ga = row["away_goals"]
        else:
            gf = row["away_goals"]
            ga = row["home_goals"]

        outcome = "W" if gf > ga else "D" if gf == ga else "L"
        results.append({
            "date": row["date"],
            "home": row["home"],
            "away": row["away"],
            "score": f"{gf}-{ga}",
            "outcome": outcome
        })
    return results

# -----------------------------
# Prediction function (unchanged)
# -----------------------------
# (keep your existing predict_match() here, unchanged)

# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def upcoming():
    load_upcoming()
    matches_by_league = {league: df_upcoming[df_upcoming["league"]==league].sort_values("datetime").head(5).to_dict(orient="records") for league in LEAGUE_ORDER}
    return render_template("upcoming.html", matches_by_league=matches_by_league)

@app.route("/match")
def match_detail():
    home = request.args.get("home")
    away = request.args.get("away")
    league = request.args.get("league")
    date = request.args.get("date")

    pred = predict_match(home, away)
    return render_template("match_detail.html", home=home, away=away, league=league, date=date, pred=pred)

# -----------------------------
if __name__ == "__main__":
    load_master()
    app.run(debug=True)
