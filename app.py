import pandas as pd
from flask import Flask, jsonify, render_template, request
from datetime import datetime
import math
import os

app = Flask(__name__)

MASTER_CSV = "master_matches.csv"
UPCOMING_CSV = "upcoming_matches.csv"

# -----------------------------
# Safe CSV Loader
# -----------------------------
def safe_load_csv(path):
    try:
        df = pd.read_csv(path, on_bad_lines="skip", encoding="utf-8")
    except:
        df = pd.read_csv(path, on_bad_lines="skip", encoding="latin1")
    df.columns = [c.lower().strip() for c in df.columns]
    for col in ["date", "home", "away"]:
        if col not in df:
            df[col] = ""
    df = df.fillna("")
    return df

# -----------------------------
# Load CSVs
# -----------------------------
df_master = safe_load_csv(MASTER_CSV)
print("MASTER LOADED:", df_master.shape)

df_upcoming = safe_load_csv(UPCOMING_CSV)
print("UPCOMING LOADED:", df_upcoming.shape)

# Normalize league names
df_upcoming["league"] = df_upcoming["league"].str.strip().str.title()

# -----------------------------
# Prediction Engine
# -----------------------------
def realistic_prediction(home, away):
    df = df_master.copy()
    for col in ["fthg","ftag"]:
        df[col] = pd.to_numeric(df.get(col,0), errors="coerce").fillna(0).astype(int)

    # Last matches
    def last_matches(team, n=10):
        home_matches = df[df.home==team]
        away_matches = df[df.away==team]
        recent = pd.concat([home_matches, away_matches]).sort_values("date", ascending=False).drop_duplicates(subset=["home","away","date"])
        return recent.head(n)

    home_recent = last_matches(home)
    away_recent = last_matches(away)

    home_exp = max(0.2, min(3, home_recent["fthg"].mean() if not home_recent.empty else 1.2))
    away_exp = max(0.2, min(3, away_recent["ftag"].mean() if not away_recent.empty else 1.0))

    # Poisson probability matrix
    def poisson_prob(lam,k):
        return math.exp(-lam) * lam**k / math.factorial(k)

    prob_matrix = [[poisson_prob(home_exp,h)*poisson_prob(away_exp,a) for a in range(6)] for h in range(6)]
    home_prob = sum(prob_matrix[h][a] for h in range(6) for a in range(6) if h>a)
    draw_prob = sum(prob_matrix[h][h] for h in range(6))
    away_prob = sum(prob_matrix[h][a] for h in range(6) for a in range(6) if h<a)

    total = home_prob + draw_prob + away_prob
    home_win = min(100, round(home_prob/total*100,1))
    draw = min(100, round(draw_prob/total*100,1))
    away_win = min(100, round(away_prob/total*100,1))

    # FT score estimate
    def xg_to_score(xg):
        if xg<0.5: return 0
        elif xg<1.2: return 1
        elif xg<2: return 2
        else: return 3
    ft_home = xg_to_score(home_exp)
    ft_away = xg_to_score(away_exp)

    btts = "YES" if ft_home>0 and ft_away>0 else "NO"
    over25 = "YES" if (ft_home+ft_away)>2 else "NO"

    # Last 4 matches with outcome on left, score on right
    def last4(team):
        home_matches = df[df.home==team]
        away_matches = df[df.away==team]
        recent = pd.concat([home_matches, away_matches]).sort_values("date", ascending=False).head(4)
        res = []
        for _, r in recent.iterrows():
            hg = int(r.get("fthg",0))
            ag = int(r.get("ftag",0))
            if r.home==team:
                score = f"{r.home} {hg}-{ag} {r.away}"
                outcome = "W" if hg>ag else "L" if hg<ag else "D"
            else:
                score = f"{r.away} {ag}-{hg} {r.home}"
                outcome = "W" if ag>hg else "L" if ag<hg else "D"
            res.append({"outcome": outcome, "score": score})
        return res

    # Logos
    def logo(team):
        folder = "static/logos"
        for f in [f"{team}.png", f"{team.lower()}.png", f"{team.replace(' ','_')}.png", f"{team.lower().replace(' ','_')}.png"]:
            if os.path.exists(os.path.join(folder,f)):
                return "/static/logos/" + f
        return "/static/logos/default.png"

    return {
        "ft_score": f"{ft_home} - {ft_away}",
        "home_win": home_win,
        "away_win": away_win,
        "draw": draw,
        "btts": btts,
        "over25": over25,
        "home_form": last4(home),
        "away_form": last4(away),
        "home_logo": logo(home),
        "away_logo": logo(away),
        "exp_home": round(home_exp,1),
        "exp_away": round(away_exp,1)
    }

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upcoming")
def api_upcoming():
    df = df_upcoming.copy()
    
    # Parse dates safely
    df["date_sort"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_sort"] = df["date_sort"].fillna(pd.Timestamp("2099-12-31"))
    
    # Normalize leagues
    df["league"] = df["league"].str.strip().str.title()
    
    # Only today or later
    today = pd.Timestamp.now()
    df_filtered = df[df["date_sort"] >= today]
    
    print("UPCOMING AFTER DATE FILTER:", df_filtered.shape)
    
    DOMINANT_LEAGUES = ["EPL", "Championship", "LaLiga", "SerieA", "Bundesliga"]
    league_priority = {l.lower(): i for i,l in enumerate(DOMINANT_LEAGUES)}
    default_priority = len(league_priority) + 1
    
    out=[]
    for league, group in df_filtered.groupby("league"):
        print(f"LEAGUE: {league}, MATCHES: {len(group)}")
        group_sorted = group.sort_values("date_sort", ascending=True).head(5)
        for _, r in group_sorted.iterrows():
            pred = realistic_prediction(r.home, r.away)
            out.append({
                "home": r.home,
                "away": r.away,
                "date": r.date,
                "time": r.get("time",""),
                "league": r.league,
                "prediction": pred,
                "home_logo": pred["home_logo"],
                "away_logo": pred["away_logo"],
                "priority": league_priority.get(r.league.lower(), default_priority)
            })
    
    out = sorted(out, key=lambda x: x["priority"])
    print("TOTAL UPCOMING MATCHES RETURNED:", len(out))
    return jsonify(out)

@app.route("/match")
def match_page():
    home = request.args.get("home")
    away = request.args.get("away")
    date = request.args.get("date")
    if not home or not away:
        return "Missing parameters",400
    pred = realistic_prediction(home, away)
    return render_template("match_details.html", home=home, away=away, date=date, pred=pred)

# -----------------------------
# Production-ready start
# -----------------------------
if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
