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
    "Man City": "Manchester City",
    "ManCity": "Manchester City",
    "Man Utd": "Manchester United",
    "MU": "Manchester United",
    "PSG": "Paris Saint-Germain",
    "Inter": "Inter Milan",
}

def normalize_team(name):
    if not name: return ""
    name = name.strip()
    # Handle title-case + aliases
    return TEAM_ALIASES.get(name, TEAM_ALIASES.get(name.title(), name))


# ------------------------------------------------------------
# SAFE CSV LOADING
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

    df["home"] = df["home"].apply(normalize_team)
    df["away"] = df["away"].apply(normalize_team)

    return df


# ------------------------------------------------------------
# MASTER DATA LOADER
# ------------------------------------------------------------
def get_master():
    global _df_master
    if _df_master is None:
        df = safe_load_csv(MASTER_CSV)

        if "fthg" in df:
            df["fthg"] = pd.to_numeric(df["fthg"], errors="coerce").fillna(0)
        if "ftag" in df:
            df["ftag"] = pd.to_numeric(df["ftag"], errors="coerce").fillna(0)

        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

        _df_master = df
        print("MASTER LOADED:", df.shape)

    return _df_master


# ------------------------------------------------------------
# UPCOMING MATCHES LOADER
# ------------------------------------------------------------
def get_upcoming():
    global _df_upcoming
    if _df_upcoming is None:
        df = safe_load_csv(UPCOMING_CSV)
        df["league"] = df["league"].str.title().fillna("Unknown")

        # Parse datetime
        def parse_dt(row):
            d = str(row["date"]).strip()
            t = str(row["time"]).strip() or "00:00"
            fmts = ["%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M", "%d/%m/%Y", "%Y-%m-%d"]
            for f in fmts:
                try:
                    return datetime.strptime(f"{d} {t}", f)
                except:
                    pass
            return None

        df["datetime"] = df.apply(parse_dt, axis=1)
        df = df[df["datetime"].notna()]

        df["home"] = df["home"].apply(normalize_team)
        df["away"] = df["away"].apply(normalize_team)

        _df_upcoming = df[["home", "away", "league", "datetime"]]
        print("UPCOMING LOADED:", _df_upcoming.shape)

    return _df_upcoming


# ------------------------------------------------------------
# LEAGUE DETECTION FOR LAST MATCHES
# ------------------------------------------------------------
def detect_league(home, away):
    df = get_master()
    rows = df[((df.home == home) & (df.away == away)) |
              ((df.home == away) & (df.away == home))]
    if len(rows) == 0:
        return None
    return rows.iloc[0].league


# ------------------------------------------------------------
# PREDICTION ENGINE
# ------------------------------------------------------------
def realistic_prediction(home, away):
    home = normalize_team(home)
    away = normalize_team(away)
    df = get_master().dropna(subset=["date"]).copy()

    # Last N matches
    def last_matches(team, n=10):
        h = df[df.home == team]
        a = df[df.away == team]
        return pd.concat([h, a]).sort_values("date", ascending=False).head(n)

    # Team stats
    def team_stats(team):
        h = df[df.home == team]
        a = df[df.away == team]
        matches = pd.concat([h, a])
        scored = h["fthg"].sum() + a["ftag"].sum()
        conceded = h["ftag"].sum() + a["fthg"].sum()
        g = len(matches)
        return {
            "avg_scored": scored / g if g else 0.8,
            "avg_conceded": conceded / g if g else 1.0
        }

    hs = team_stats(home)
    aw = team_stats(away)

    home_exp = 0.6 * hs["avg_scored"] + 0.4 * aw["avg_conceded"]
    away_exp = 0.6 * aw["avg_scored"] + 0.4 * hs["avg_conceded"]

    home_exp *= 1.1  # home advantage

    # Poisson model
    def pois(lam, k):
        return math.exp(-lam) * lam**k / math.factorial(k)

    home_p = sum(pois(home_exp, h) * pois(away_exp, a) for h in range(6) for a in range(6) if h > a)
    draw_p = sum(pois(home_exp, h) * pois(away_exp, h) for h in range(6))
    away_p = sum(pois(home_exp, h) * pois(away_exp, a) for h in range(6) for a in range(6) if h < a)

    total = home_p + draw_p + away_p
    home_pct = round(home_p / total * 100, 1)
    draw_pct = round(draw_p / total * 100, 1)
    away_pct = round(away_p / total * 100, 1)

    # Expected goal → rounded score
    def xg_to_score(x):
        if x < 0.5: return 0
        if x < 1.2: return 1
        if x < 2.0: return 2
        return 3

    ft_h = xg_to_score(home_exp)
    ft_a = xg_to_score(away_exp)
    btts = "YES" if (ft_h > 0 and ft_a > 0) else "NO"
    over25 = "YES" if (ft_h + ft_a) > 2 else "NO"

    # ---- LAST 4 MATCHES (LEAGUE FILTERED) ----
    league = detect_league(home, away)
    df_l = df[df["league"].str.lower() == league.lower()] if league else df

    def last4(team):
        h = df_l[df_l.home == team]
        a = df_l[df_l.away == team]
        r = pd.concat([h, a]).sort_values("date", ascending=False).head(4)

        out = []
        for _, row in r.iterrows():
            hg, ag = int(row.fthg), int(row.ftag)
            d = row.date.strftime("%d/%m/%Y") if pd.notna(row.date) else ""
            if row.home == team:
                outcome = "W" if hg > ag else "L" if hg < ag else "D"
                score = f"{row.home} {hg}-{ag} {row.away} ({d})"
            else:
                outcome = "W" if ag > hg else "L" if ag < hg else "D"
                score = f"{row.away} {ag}-{hg} {row.home} ({d})"
            out.append({"outcome": outcome, "score": score})
        return out

    return {
        "ft_score": f"{ft_h} - {ft_a}",
        "home_win": home_pct,
        "draw": draw_pct,
        "away_win": away_pct,
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

    # TRUE GLOBAL CHRONOLOGY
    df = df.sort_values("datetime")

    out = []
    for _, r in df.iterrows():
        pred = realistic_prediction(r.home, r.away)
        out.append({
            "home": r.home,
            "away": r.away,
            "date": r.datetime.strftime("%Y-%m-%d"),
            "time": r.datetime.strftime("%H:%M"),
            "league": r.league,
            "prediction": pred
        })

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
# RUN APP
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
