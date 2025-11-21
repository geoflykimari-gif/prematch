import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime

app = Flask(__name__)

MASTER_CSV = "master_matches.csv"
UPCOMING_CSV = "upcoming_matches.csv"

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
# Team normalization
# -----------------------------
TEAM_ALIASES = {
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
    # add all other aliases
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

    # Strip spaces from column names
    df_master.columns = [c.strip() for c in df_master.columns]

    # Fill missing time
    if "time" not in df_master.columns:
        df_master["time"] = "00:00"
    else:
        df_master["time"] = df_master["time"].fillna("00:00")

    # Combine date and time
    df_master["datetime"] = pd.to_datetime(
        df_master["date"].astype(str) + " " + df_master["time"].astype(str),
        errors="coerce"
    )
    df_master["date"] = df_master["datetime"].dt.date.astype(str)

    # Rename goal columns
    if "fthg" in df_master.columns:
        df_master.rename(columns={"fthg": "home_goals"}, inplace=True)
    if "ftag" in df_master.columns:
        df_master.rename(columns={"ftag": "away_goals"}, inplace=True)

    # Convert goals to integers
    df_master["home_goals"] = pd.to_numeric(df_master["home_goals"], errors="coerce").fillna(0).astype(int)
    df_master["away_goals"] = pd.to_numeric(df_master["away_goals"], errors="coerce").fillna(0).astype(int)

    # Normalize team names
    df_master["home"] = df_master["home"].apply(normalize_team)
    df_master["away"] = df_master["away"].apply(normalize_team)

# -----------------------------
# Load upcoming CSV
# -----------------------------
def load_upcoming():
    global df_upcoming
    df_upcoming = pd.read_csv(UPCOMING_CSV)
    df_upcoming.columns = [c.strip() for c in df_upcoming.columns]

    df_upcoming["datetime"] = pd.to_datetime(df_upcoming["datetime"], errors="coerce")
    df_upcoming["date"] = df_upcoming["datetime"].dt.date.astype(str)
    df_upcoming["time"] = df_upcoming["datetime"].dt.strftime("%H:%M")

    # Normalize team names
    df_upcoming["home"] = df_upcoming["home"].apply(normalize_team)
    df_upcoming["away"] = df_upcoming["away"].apply(normalize_team)

# -----------------------------
# Last N matches function
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
# Strong football-like prediction
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

    # Average goals
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

    # Football-like power
    home_power = home_score*0.7 + home_gf*0.5 - home_ga*0.3
    away_power = away_score*0.7 + away_gf*0.5 - away_ga*0.3
    home_power *= 1.1

    total = home_power + away_power + 0.01
    home_win = round(home_power/total*100,1)
    away_win = round(away_power/total*100,1)
    draw = round(100 - home_win - away_win,1)

    # Full-time score prediction
    predicted_gf = (home_gf + away_ga*0.8)/2
    predicted_ga = (away_gf + home_ga*0.8)/2
    if home_win > away_win:
        ft_score = f"{max(1,round(predicted_gf))}-{max(0,round(predicted_ga))}"
    elif away_win > home_win:
        ft_score = f"{max(1,round(predicted_ga))}-{max(0,round(predicted_gf))}"
    else:
        ft_score = "1-1"

    btts = "Likely" if home_gf+away_gf>2.5 else "Unlikely"
    over25 = "Likely" if predicted_gf+predicted_ga>2.5 else "Unlikely"

    return {
        "prediction":{"home_win":f"{home_win}%", "draw":f"{draw}%", "away_win":f"{away_win}%"},
        "home_form": home_form[:4],  # last 4 for UI
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
    for league in LEAGUE_ORDER:
        df_l = df_upcoming[df_upcoming["league"]==league].sort_values("datetime").head(5)
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
