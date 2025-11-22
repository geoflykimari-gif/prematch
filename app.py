from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime
import numpy as np

app = Flask(__name__)

# CSV paths
MASTER_CSV = "master_matches.csv"
UPCOMING_CSV = "upcoming_matches.csv"

LEAGUE_ORDER = ["EPL","Championship","La Liga","Serie A","Bundesliga","Ligue 1"]

# --- CSV Loader ---
def load_csv(path, is_master=False):
    df = pd.read_csv(path)

    # Detect date column
    date_col = None
    for col in ['datetime','date']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"CSV {path} must have 'datetime' or 'date' column")

    # Strip strings
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()

    # Lowercase team names
    df['home'] = df['home'].str.lower()
    df['away'] = df['away'].str.lower()
    if 'league' in df.columns:
        df['league'] = df['league'].str.strip()

    # Parse datetime safely
    df[date_col] = pd.to_datetime(df[date_col].str.strip(), errors='coerce')

    # Rename to datetime
    if date_col != 'datetime':
        df.rename(columns={date_col:'datetime'}, inplace=True)

    # Ensure numeric goals
    if is_master:
        df['fthg'] = pd.to_numeric(df.get('fthg',0), errors='coerce').fillna(0).astype(int)
        df['ftag'] = pd.to_numeric(df.get('ftag',0), errors='coerce').fillna(0).astype(int)

    return df

df_master = load_csv(MASTER_CSV, is_master=True)
df_upcoming = load_csv(UPCOMING_CSV)

# --- Normalize teams ---
def normalize_teams(df):
    df['home'] = df['home'].str.strip().str.lower()
    df['away'] = df['away'].str.strip().str.lower()
    return df

df_master = normalize_teams(df_master)
df_upcoming = normalize_teams(df_upcoming)

# --- Helpers ---
def get_last_n_matches(team, n=5):
    team = team.lower().strip()
    df = df_master[(df_master['home']==team)|(df_master['away']==team)]
    df = df.sort_values('datetime', ascending=False).head(n)
    matches = []
    for _, row in df.iterrows():
        opponent = row['away'] if row['home']==team else row['home']
        venue = 'Home' if row['home']==team else 'Away'

        home_goals = int(row.get('fthg',0))
        away_goals = int(row.get('ftag',0))

        score = f"{home_goals}-{away_goals}" if venue=='Home' else f"{away_goals}-{home_goals}"

        if venue=='Home':
            result = 'W' if home_goals>away_goals else 'D' if home_goals==away_goals else 'L'
        else:
            result = 'W' if away_goals>home_goals else 'D' if away_goals==home_goals else 'L'

        matches.append({
            'date': row['datetime'].strftime('%Y-%m-%d %H:%M'),
            'opponent': opponent.title(),
            'venue': venue,
            'score': score,
            'result': result
        })
    return matches

def get_head_to_head(team1, team2, n=5):
    team1 = team1.lower().strip()
    team2 = team2.lower().strip()
    df = df_master[((df_master['home']==team1)&(df_master['away']==team2))|
                   ((df_master['home']==team2)&(df_master['away']==team1))]
    df = df.sort_values('datetime', ascending=False).head(n)
    matches = []
    for _, row in df.iterrows():
        score = f"{row['fthg']}-{row['ftag']}"
        if row['home']==team1:
            result = 'W' if row['fthg']>row['ftag'] else 'D' if row['fthg']==row['ftag'] else 'L'
        else:
            result = 'W' if row['ftag']>row['fthg'] else 'D' if row['ftag']==row['fthg'] else 'L'
        matches.append({
            'date': row['datetime'].strftime('%Y-%m-%d %H:%M'),
            'home': row['home'].title(),
            'away': row['away'].title(),
            'score': score,
            'result': result
        })
    return matches

# --- Prediction ---
def get_match_prediction(home, away):
    last_home = get_last_n_matches(home, 5)
    last_away = get_last_n_matches(away, 5)

    home_avg = np.mean([int(m['score'].split('-')[0]) for m in last_home]) if last_home else 1
    away_avg = np.mean([int(m['score'].split('-')[1]) for m in last_away]) if last_away else 1

    exp_home_goals = round(home_avg)
    exp_away_goals = round(away_avg)
    predicted_score = f"{exp_home_goals}-{exp_away_goals}"

    home_pct = min(max(40 + (exp_home_goals-exp_away_goals)*10,5),90)
    draw_pct = min(max(100 - home_pct - 10,5),50)
    away_pct = 100 - home_pct - draw_pct

    total_goals = exp_home_goals + exp_away_goals
    over_25 = 70 if total_goals>2 else 30
    under_25 = 100 - over_25
    btts = 70 if exp_home_goals>0 and exp_away_goals>0 else 30

    return {
        'home_pct': int(home_pct),
        'draw_pct': int(draw_pct),
        'away_pct': int(away_pct),
        'predicted_score': predicted_score,
        'exp_home_goals': exp_home_goals,
        'exp_away_goals': exp_away_goals,
        'btts_pct': btts,
        'over_25_pct': over_25,
        'under_25_pct': under_25
    }

# --- Routes ---
@app.route("/")
def index():
    now = datetime.now()
    df_up = df_upcoming[df_upcoming['datetime']>=now].copy()
    df_up['league_order'] = df_up['league'].apply(lambda x: LEAGUE_ORDER.index(x) if x in LEAGUE_ORDER else 99)
    df_up = df_up.sort_values(['league_order','datetime'])
    matches_by_league = {}
    for league in LEAGUE_ORDER:
        league_matches = df_up[df_up['league']==league].head(5)
        if not league_matches.empty:
            matches_by_league[league] = league_matches.to_dict('records')
    return render_template("index.html", matches_by_league=matches_by_league)

@app.route("/match")
def match_detail():
    home_team = request.args.get('home','').strip().lower()
    away_team = request.args.get('away','').strip().lower()
    league = request.args.get('league','')
    match_date = request.args.get('date','TBD')

    if not home_team or not away_team:
        return "Missing home or away team",400

    pred = get_match_prediction(home_team, away_team)
    last_home = get_last_n_matches(home_team,5)
    last_away = get_last_n_matches(away_team,5)
    h2h = get_head_to_head(home_team, away_team,5)

    return render_template(
        "match_detail.html",
        home_team=home_team.title(),
        away_team=away_team.title(),
        league=league,
        match_date=match_date,
        pred=pred,
        last_home=last_home,
        last_away=last_away,
        h2h=h2h
    )

@app.route("/info")
def info():
    return render_template("info.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

# Optional: simple ping route to keep Render awake
@app.route("/ping")
def ping():
    return "pong"

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
