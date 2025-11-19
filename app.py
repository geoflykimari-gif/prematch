import pandas as pd
from flask import Flask, jsonify, render_template, request
from datetime import datetime
import math
import os

app = Flask(__name__)

MASTER_CSV = "master_matches.csv"
UPCOMING_CSV = "upcoming_matches.csv"

_df_master = None
_df_upcoming = None

# ------------------------------------------------------------
# TEAM ALIASES
# ------------------------------------------------------------
TEAM_ALIASES = {
    # Championship
    "Preston": "Preston North End",
    "Bristol City FC": "Bristol City",
    "Charlton": "Charlton Athletic",
    "Sheffield Wed": "Sheffield Wednesday",
    "Blackburn": "Blackburn Rovers",
    "Middlesbrough FC": "Middlesbrough",
    "Portsmouth FC": "Portsmouth",
    "Hull": "Hull City",
    # EPL examples
    "Man City": "Manchester City",
    "ManCity": "Manchester City",
    "Man Utd": "Manchester United",
    "MU": "Manchester United",
    "PSG": "Paris Saint-Germain",
    "Inter": "Inter Milan",
}

def normalize_team(name):
    if not name:
        return ""
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name.title())

# ------------------------------------------------------------
# CSV LOADING
# ------------------------------------------------------------
def safe_load_csv(path):
    try:
        df = pd.read_csv(path, on_bad_lines="skip", encoding="utf-8")
    except:
        df = pd.read_csv(path, on_bad_lines="skip", encoding="latin1")

    df.columns = [c.lower().strip() for c in df.columns]
    for col in ["date", "time", "home", "away", "league"]:
        if col not in df:
            df[col] = ""
    df = df.fillna("")
    return df

def get_master():
    global _df_master
    if _df_master is None:
        _df_master = safe_load_csv(MASTER_CSV)
        for col in ["fthg", "ftag"]:
            if col in _df_master:
                _df_master[col] = pd.to_numeric(_df_master[col], errors="coerce").fillna(0).astype(float)
        if "date" in _df_master:
            _df_master["date"] = pd.to_datetime(_df_master["date"], errors="coerce")
        # Normalize team names
        _df_master["home"] = _df_master["home"].apply(normalize_team)
        _df_master["away"] = _df_master["away"].apply(normalize_team)
        print("MASTER LOADED:", _df_master.shape)
    return _df_master

def get_upcoming():
    global _df_upcoming
    if _df_upcoming is None:
        df = safe_load_csv(UPCOMING_CSV)
        df["league"] = df["league"].str.title().fillna("Unknown")

        def parse_datetime(row):
            date_str = str(row.get("date", "")).strip()
            time_str = str(row.get("time", "")).strip() or "00:00"
            formats = ["%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d", "%d/%m/%Y"]
            for fmt in formats:
                try:
                    return pd.to_datetime(f"{date_str} {time_str}", format=fmt)
                except:
                    continue
            return None

        df["datetime"] = df.apply(parse_datetime, axis=1)
        df = df[df["datetime"].notna()]

        # Normalize team names
        df["home"] = df["home"].apply(normalize_team)
        df["away"] = df["away"].apply(normalize_team)

        _df_upcoming = df[["home", "away", "league", "datetime"]]
        print(f"UPCOMING LOADED: {_df_upcoming.shape}")
    return _df_upcoming

# ------------------------------------------------------------
# LEAGUE ORDER
# ------------------------------------------------------------
LEAGUE_ORDER = [
    "Epl", "Championship", "La Liga", "Serie A", "Ligue 1",
    "Bundesliga", "Eredivisie", "Scottish Premiership",
    "Turkish Super Lig", "Primera División", "Argentina Primera"
]

def league_rank(league):
    try:
        return LEAGUE_ORDER.index(str(league).strip())
    except ValueError:
        return 999

# ------------------------------------------------------------
# PERFORMANCE-BASED PREDICTION ENGINE
# ------------------------------------------------------------
def realistic_prediction(home, away):
    home = normalize_team(home)
    away = normalize_team(away)
    df = get_master().copy()
    df = df.dropna(subset=["date"])

    # ---- Last N matches for form ----
    def last_matches(team, n=10):
        home_m = df[df.home == team]
        away_m = df[df.away == team]
        recent = pd.concat([home_m, away_m]).sort_values("date", ascending=False)
        return recent.head(n)

    # ---- Season performance stats ----
    def team_stats(team):
        home_m = df[df.home == team]
        away_m = df[df.away == team]
        matches = pd.concat([home_m, away_m])
        scored = home_m["fthg"].sum() + away_m["ftag"].sum()
        conceded = home_m["ftag"].sum() + away_m["fthg"].sum()
        games = len(matches)
        return {
            "avg_scored": scored / games if games else 0.8,
            "avg_conceded": conceded / games if games else 1.0
        }

    home_stats = team_stats(home)
    away_stats = team_stats(away)

    # ---- Expected goals ----
    home_exp = 0.6 * home_stats["avg_scored"] + 0.4 * away_stats["avg_conceded"]
    away_exp = 0.6 * away_stats["avg_scored"] + 0.4 * home_stats["avg_conceded"]

    # Optional: home advantage
    home_exp *= 1.1

    # ---- Poisson probability ----
    def poisson_prob(lam, k):
        return math.exp(-lam) * lam**k / math.factorial(k)

    prob_matrix = [
        [poisson_prob(home_exp, h) * poisson_prob(away_exp, a) for a in range(6)]
        for h in range(6)
    ]

    home_prob = sum(prob_matrix[h][a] for h in range(6) for a in range(6) if h > a)
    draw_prob = sum(prob_matrix[h][h] for h in range(6))
    away_prob = sum(prob_matrix[h][a] for h in range(6) for a in range(6) if h < a)

    total = home_prob + draw_prob + away_prob
    home_win = round(home_prob / total * 100, 1)
    draw = round(draw_prob / total * 100, 1)
    away_win = round(away_prob / total * 100, 1)

    # ---- Full-time predicted score (rounded from expected goals) ----
    def xg_to_score(xg):
        if xg < 0.5: return 0
        elif xg < 1.2: return 1
        elif xg < 2: return 2
        else: return 3

    ft_home = xg_to_score(home_exp)
    ft_away = xg_to_score(away_exp)
    btts = "YES" if ft_home > 0 and ft_away > 0 else "NO"
    over25 = "YES" if (ft_home + ft_away) > 2 else "NO"

    # ---- Last 4 matches
    def last4(team):
        recent = last_matches(team, n=4)
        res = []
        for _, r in recent.iterrows():
            hg, ag = int(r.fthg), int(r.ftag)
            match_date = r.date.strftime("%d/%m/%Y") if pd.notna(r.date) else ""
            if r.home == team:
                outcome = "W" if hg>ag else "L" if hg<ag else "D"
                score = f"{r.home} {hg}-{ag} {r.away} ({match_date})"
            else:
                outcome = "W" if ag>hg else "L" if ag<hg else "D"
                score = f"{r.away} {ag}-{hg} {r.home} ({match_date})"
            res.append({"outcome": outcome, "score": score})
        return res

    return {
        "ft_score": f"{ft_home} - {ft_away}",
        "home_win": home_win,
        "away_win": away_win,
        "draw": draw,
        "btts": btts,
        "over25": over25,
        "home_form": last4(home),
        "away_form": last4(away),
    }

# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upcoming")
def api_upcoming():
    df = get_upcoming().copy()
    today = pd.Timestamp.now().normalize()
    df = df[df["datetime"].dt.normalize() >= today]

    out = []
    for league, group in df.groupby("league"):
        group_sorted = group.sort_values("datetime").head(5)
        for _, r in group_sorted.iterrows():
            pred = realistic_prediction(r.home, r.away)
            out.append({
                "home": r.home,
                "away": r.away,
                "date": r.datetime.strftime("%Y-%m-%d"),
                "time": r.datetime.strftime("%H:%M") if r.datetime.time() != datetime.min.time() else "",
                "league": league,
                "prediction": pred
            })

    out = sorted(out, key=lambda x: (league_rank(x["league"]), x["date"], x["time"]))
    return jsonify(out)

@app.route("/match")
def match_page():
    home = request.args.get("home")
    away = request.args.get("away")
    date = request.args.get("date")
    if not home or not away:
        return "Missing parameters", 400

    pred = realistic_prediction(home, away)
    return render_template("match_details.html", home=home, away=away, date=date, pred=pred)

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
