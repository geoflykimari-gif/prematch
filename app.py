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

def safe_load_csv(path):
    try:
        df = pd.read_csv(path, on_bad_lines="skip", encoding="utf-8")
    except:
        df = pd.read_csv(path, on_bad_lines="skip", encoding="latin1")
    df.columns = [c.lower().strip() for c in df.columns]
    for col in ["date", "home", "away", "league"]:
        if col not in df:
            df[col] = ""
    df = df.fillna("")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def get_master():
    global _df_master
    if _df_master is None:
        _df_master = safe_load_csv(MASTER_CSV)
        print("MASTER LOADED:", _df_master.shape)
    return _df_master


def get_upcoming():
    global _df_upcoming
    if _df_upcoming is None:
        _df_upcoming = safe_load_csv(UPCOMING_CSV)
        _df_upcoming["league"] = _df_upcoming["league"].str.title()
        print("UPCOMING LOADED:", _df_upcoming.shape)
    return _df_upcoming

# Prediction Engine
def realistic_prediction(home, away):
    df = get_master().copy()
    for col in ["fthg", "ftag"]:
        df[col] = pd.to_numeric(df.get(col,0), errors="coerce").fillna(0).astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    def last_matches(team, n=10):
        home_m = df[df.home==team]
        away_m = df[df.away==team]
        recent = pd.concat([home_m, away_m]).sort_values("date", ascending=False)
        return recent.head(n)

    home_recent = last_matches(home)
    away_recent = last_matches(away)

    home_exp = max(0.2, min(3, home_recent["fthg"].mean() if not home_recent.empty else 1.2))
    away_exp = max(0.2, min(3, away_recent["ftag"].mean() if not away_recent.empty else 1.0))

    def poisson_prob(lam,k):
        return math.exp(-lam) * lam**k / math.factorial(k)

    prob_matrix = [[poisson_prob(home_exp,h)*poisson_prob(away_exp,a) for a in range(6)] for h in range(6)]
    home_prob = sum(prob_matrix[h][a] for h in range(6) for a in range(6) if h>a)
    draw_prob = sum(prob_matrix[h][h] for h in range(6))
    away_prob = sum(prob_matrix[h][a] for h in range(6) for a in range(6) if h<a)

    total = home_prob + draw_prob + away_prob
    home_win = round(home_prob/total*100,1)
    draw = round(draw_prob/total*100,1)
    away_win = round(away_prob/total*100,1)

    def xg_to_score(xg):
        if xg<0.5: return 0
        elif xg<1.2: return 1
        elif xg<2: return 2
        else: return 3

    ft_home = xg_to_score(home_exp)
    ft_away = xg_to_score(away_exp)

    btts = "YES" if ft_home>0 and ft_away>0 else "NO"
    over25 = "YES" if (ft_home+ft_away)>2 else "NO"

    def last4(team):
        home_m = df[df.home==team]
        away_m = df[df.away==team]
        recent = pd.concat([home_m, away_m]).sort_values("date", ascending=False).head(4)
        res = []
        for _, r in recent.iterrows():
            hg, ag = int(r.fthg), int(r.ftag)
            if r.home==team:
                outcome = "W" if hg>ag else "L" if hg<ag else "D"
                score = f"{r.home} {hg}-{ag} {r.away}"
            else:
                outcome = "W" if ag>hg else "L" if ag<hg else "D"
                score = f"{r.away} {ag}-{hg} {r.home}"
            res.append({"outcome": outcome, "score": score})
        return res

    def logo(team):
        folder = "static/logos"
        team_variants = [
            f"{team}.png",
            f"{team.lower()}.png",
            f"{team.title()}.png",
            f"{team.replace(' ','_')}.png",
            f"{team.lower().replace(' ','_')}.png",
            f"{team.title().replace(' ','_')}.png",
            f"{team} FC.png",
            f"{team.lower()} fc.png",
            f"{team.title()} FC.png",
            f"{team.replace(' ','_')} FC.png",
            f"{team.lower().replace(' ','_')} fc.png",
            f"{team.title().replace(' ','_')} FC.png",
        ]
        for f in team_variants:
            if os.path.exists(os.path.join(folder, f)):
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
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upcoming")
def api_upcoming():
    df = get_upcoming().copy()
    today = pd.Timestamp.now()
    df = df[df["date"] >= today]

    out=[]
    for league, group in df.groupby("league"):
        group_sorted = group.sort_values("date").head(5)
        for _, r in group_sorted.iterrows():
            pred = realistic_prediction(r.home,r.away)
            out.append({
                "home": r.home,
                "away": r.away,
                "date": r.date,
                "time": r.get("time",""),
                "league": league,
                "prediction": pred,
                "home_logo": pred["home_logo"],
                "away_logo": pred["away_logo"]
            })
    return jsonify(out)

@app.route("/match")
def match_page():
    home = request.args.get("home")
    away = request.args.get("away")
    date = request.args.get("date")
    if not home or not away:
        return "Missing parameters",400
    pred = realistic_prediction(home,away)
    return render_template("match_details.html", home=home, away=away, date=date, pred=pred)

if __name__=="__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)
