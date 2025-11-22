import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime
import os
import numpy as np

app = Flask(__name__)

# -----------------------------
# File paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(BASE_DIR, "master_matches.csv")
UPCOMING_CSV = os.path.join(BASE_DIR, "upcoming_matches.csv")

df_master = None
df_upcoming = None

# -----------------------------
# League order
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
# Team normalization
# -----------------------------
TEAM_ALIASES = {
    "Man City": "Manchester City",
    "ManCity": "Manchester City",
    "Manchester City FC": "Manchester City",
    "Man Utd": "Manchester United",
    "Man United": "Manchester United",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Arsenal FC": "Arsenal",
    # Add more aliases if needed
}

def normalize_team(name):
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)

# -----------------------------
# Load master CSV
# -----------------------------
def load_master():
    global df_master
    df_master = pd.read_csv(MASTER_CSV)
    df_master.columns = [c.strip() for c in df_master.columns]
    df_master.rename(columns={"fthg": "home_goals", "ftag": "away_goals"}, inplace=True)
    if "time" in df_master.columns:
        df_master["time"] = df_master["time"].fillna("00:00")
        df_master["datetime"] = pd.to_datetime(
            df_master["date"].astype(str) + " " + df_master["time"].astype(str),
            errors="coerce"
        )
    else:
        df_master["datetime"] = pd.to_datetime(df_master["date"], errors="coerce")
        df_master["time"] = "00:00"
    df_master["date"] = df_master["datetime"].dt.date.astype(str)
    df_master["home_goals"] = pd.to_numeric(df_master["home_goals"], errors="coerce").fillna(0).astype(int)
    df_master["away_goals"] = pd.to_numeric(df_master["away_goals"], errors="coerce").fillna(0).astype(int)
    df_master["home"] = df_master["home"].apply(normalize_team)
    df_master["away"] = df_master["away"].apply(normalize_team)

# -----------------------------
# Load upcoming CSV
# -----------------------------
def load_upcoming():
    global df_upcoming
    if not os.path.exists(UPCOMING_CSV):
        raise FileNotFoundError(f"Upcoming CSV not found: {UPCOMING_CSV}")
    df_upcoming = pd.read_csv(UPCOMING_CSV)
    df_upcoming.columns = [c.strip() for c in df_upcoming.columns]
    if "datetime" not in df_upcoming.columns:
        date_col = next((c for c in ["date", "Date", "match_date"] if c in df_upcoming.columns), None)
        time_col = next((c for c in ["time", "Time", "match_time"] if c in df_upcoming.columns), None)
        if not date_col:
            raise KeyError("No date column found in upcoming CSV")
        if not time_col:
            df_upcoming["time"] = "00:00"
            time_col = "time"
        df_upcoming["datetime"] = pd.to_datetime(
            df_upcoming[date_col].astype(str) + " " + df_upcoming[time_col].astype(str),
            errors="coerce"
        )
    else:
        df_upcoming["datetime"] = pd.to_datetime(df_upcoming["datetime"], errors="coerce")
    df_upcoming["date"] = df_upcoming["datetime"].dt.date.astype(str)
    df_upcoming["time"] = df_upcoming["datetime"].dt.strftime("%H:%M")
    df_upcoming["home"] = df_upcoming["home"].apply(normalize_team)
    df_upcoming["away"] = df_upcoming["away"].apply(normalize_team)

# -----------------------------
# Last N matches
# -----------------------------
def get_last_n_matches(team, n=10):
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
# Predict match
# -----------------------------
def predict_match(home, away):
    home_form = get_last_n_matches(home, 10)
    away_form = get_last_n_matches(away, 10)

    def weighted_score(form):
        score_map = {"W":3, "D":1, "L":0}
        weights = [1.5,1.4,1.3,1.2,1.1,1,1,1,1,1][:len(form)]
        return sum(score_map[m["outcome"]]*w for m,w in zip(form,weights))

    home_score = weighted_score(home_form)
    away_score = weighted_score(away_form)

    def avg_goals(team):
        matches = get_last_n_matches(team,10)
        gf=ga=0
        for m in matches:
            if m["home"]==team:
                gf += int(m["score"].split("-")[0])
                ga += int(m["score"].split("-")[1])
            else:
                gf += int(m["score"].split("-")[1])
                ga += int(m["score"].split("-")[0])
        n=len(matches)
        return (gf/n if n>0 else 1.0, ga/n if n>0 else 1.0)

    home_gf, home_ga = avg_goals(home)
    away_gf, away_ga = avg_goals(away)

    home_power = home_score*0.7 + home_gf*0.6 - home_ga*0.4
    away_power = away_score*0.7 + away_gf*0.6 - away_ga*0.4
    home_power *= 1.1

    total = home_power + away_power + 0.01
    home_win = round(home_power/total*100,1)
    away_win = round(away_power/total*100,1)
    draw = round(100 - home_win - away_win,1)

    # -----------------------------
    # Minimal changes: integer goals & realistic BTTS / Over 2.5
    # -----------------------------
    # Expected goals using adjusted averages
    home_exp = (home_gf + away_ga*0.9)/2
    away_exp = (away_gf + home_ga*0.9)/2

    # Sample goals from Poisson for realism
    home_goals_pred = np.random.poisson(max(0.5, home_exp))
    away_goals_pred = np.random.poisson(max(0.5, away_exp))

    # BTTS probability
    btts = "Likely" if home_goals_pred>0 and away_goals_pred>0 else "Unlikely"
    # Over 2.5 goals
    over25 = "Likely" if (home_goals_pred + away_goals_pred) > 2 else "Unlikely"

    ft_score = f"{home_goals_pred}-{away_goals_pred}"

    return {
        "prediction":{"home_win":f"{home_win}%", "draw":f"{draw}%", "away_win":f"{away_win}%"},
        "home_form": home_form[:4],
        "away_form": away_form[:4],
        "btts":btts,
        "over25":over25,
        "ft_score":ft_score
    }

# -----------------------------
# Order upcoming matches
# -----------------------------
def order_upcoming():
    grouped = {}
    today = datetime.today().date()
    for league in LEAGUE_ORDER:
        df_l = df_upcoming[(df_upcoming["league"]==league) & 
                           (df_upcoming["datetime"].dt.date >= today)
                          ].sort_values("datetime").head(4)
        grouped[league] = df_l.to_dict(orient="records")
    return grouped

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def upcoming():
    load_upcoming()
    matches_by_league = order_upcoming()
    return render_template("upcoming.html", matches_by_league=matches_by_league)

@app.route("/match")
def match_detail():
    home = request.args.get("home")
    away = request.args.get("away")
    league = request.args.get("league")
    date = request.args.get("date")
    pred = predict_match(home, away)
    return render_template(
        "match_detail.html",
        home=home,
        away=away,
        league=league,
        date=date,
        pred=pred
    )

# -----------------------------
if __name__ == "__main__":
    load_master()
    app.run(debug=True)
