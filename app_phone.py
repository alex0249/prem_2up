import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import pickle
import datetime
import joblib
import os
import warnings
from sklearn.linear_model import Ridge
from fuzzywuzzy import process

warnings.filterwarnings('ignore')

# ==========================================
# 🔑 CONFIGURATION
# ==========================================
st.set_page_config(page_title="2-Up Master Suite", layout="centered")

# API Configuration
API_KEY = 'c46720c50cc04c499506a757b9a00259' 
BASE_URL_API = 'https://api.football-data.org/v4/matches'
LEAGUE_CODES_API = ['PL', 'PD', 'BL1', 'SA', 'FL1']

# Training Data Configuration
SEASONS_TO_FETCH = ['2425', '2526'] 
BASE_URL_DATA = "https://www.football-data.co.uk/mmz4281"
LEAGUE_MAP = {'E0': 'PL', 'SP1': 'PD', 'D1': 'BL1', 'I1': 'SA', 'F1': 'FL1'}

# Live Predictor Configuration (Tab 1)
LEAGUES_LIVE = ['EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1']

# ==========================================
# ⚙️ AUTOMATED BACKEND: DATA & TRAINING
# ==========================================

@st.cache_data(ttl=86400)
def get_latest_stats_and_model():
    """Automates the download, processing, and training cycle."""
    all_data = []
    
    # 1. Fetch CSV Data
    for s in SEASONS_TO_FETCH:
        for csv_code, api_code in LEAGUE_MAP.items():
            url = f"{BASE_URL_DATA}/{s}/{csv_code}.csv"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    df = pd.read_csv(io.StringIO(response.content.decode('latin1')))
                    if 'Date' in df.columns and 'HomeTeam' in df.columns:
                        df['League_ID'] = api_code 
                        all_data.append(df)
            except: continue

    if not all_data:
        return None, None, "❌ Failed to download historical data."

    full_df = pd.concat(all_data, ignore_index=True)
    
    # 2. Process Features
    req_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HST', 'AST', 'HC', 'AC', 'League_ID']
    df = full_df[full_df.columns.intersection(req_cols)].dropna()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date')
    
    home_stats = df[['Date', 'HomeTeam', 'League_ID', 'FTHG', 'FTAG', 'HST', 'HC']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GF', 'FTAG': 'GA', 'HST': 'SOT', 'HC': 'Corn'}
    )
    away_stats = df[['Date', 'AwayTeam', 'League_ID', 'FTAG', 'FTHG', 'AST', 'AC']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GF', 'FTHG': 'GA', 'AST': 'SOT', 'AC': 'Corn'}
    )
    
    team_df = pd.concat([home_stats, away_stats]).sort_values('Date')
    
    for c in ['GF', 'GA', 'SOT', 'Corn']:
        team_df[f'avg_{c}'] = team_df.groupby('Team')[c].transform(
            lambda x: x.rolling(5, closed='left').mean()
        )
    
    latest_stats = team_df.groupby('Team').last().reset_index()
    
    # 3. Build Training Set
    train_df = df.merge(team_df[['Date', 'Team', 'avg_GF', 'avg_GA', 'avg_SOT', 'avg_Corn']], 
                      left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='inner')
    train_df = train_df.rename(columns={'avg_GF': 'H_Att', 'avg_GA': 'H_Def', 'avg_SOT': 'H_SOT', 'avg_Corn': 'H_Corn'}).drop(columns=['Team'])
    
    train_df = train_df.merge(team_df[['Date', 'Team', 'avg_GF', 'avg_GA', 'avg_SOT', 'avg_Corn']], 
                      left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='inner')
    train_df = train_df.rename(columns={'avg_GF': 'A_Att', 'avg_GA': 'A_Def', 'avg_SOT': 'A_SOT', 'avg_Corn': 'A_Corn'}).drop(columns=['Team'])
    
    train_df['total_goals'] = train_df['FTHG'] + train_df['FTAG']
    train_df = train_df.dropna()

    # 4. Train Model
    features = ['H_Att', 'H_Def', 'H_SOT', 'H_Corn', 'A_Att', 'A_Def', 'A_SOT', 'A_Corn']
    model = Ridge(alpha=10)
    model.fit(train_df[features], train_df['total_goals'])
    
    return model, latest_stats, "Success"

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def load_live_system(league_name):
    """Loads specific models for Tab 1 (requires local files)."""
    m_file, s_file = f"model_{league_name}.pkl", f"scaler_{league_name}.pkl"
    if not os.path.exists(m_file) or not os.path.exists(s_file):
        raise FileNotFoundError(f"⚠️ Missing {m_file} or {s_file}")
    return joblib.load(m_file), joblib.load(s_file)

def get_fixtures_api():
    headers = {'X-Auth-Token': API_KEY}
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    next_week = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    params = {'dateFrom': today, 'dateTo': next_week, 'competitions': ','.join(LEAGUE_CODES_API), 'status': 'SCHEDULED'}
    
    try:
        r = requests.get(BASE_URL_API, headers=headers, params=params)
        matches = r.json().get('matches', [])
        return pd.DataFrame([{
            'Date': m['utcDate'][:10], 'League_Code': m['competition']['code'],
            'Home': m['homeTeam']['name'], 'Away': m['awayTeam']['name']
        } for m in matches])
    except: return pd.DataFrame()

def strict_match(team_name, league_code, stats_db):
    league_teams = stats_db[stats_db['League_ID'] == league_code]
    if league_teams.empty: return None
    choices = league_teams['Team'].unique()
    match, score = process.extractOne(team_name, choices)
    return league_teams[league_teams['Team'] == match].iloc[0] if score > 65 else None

def run_analysis_pipeline():
    model, stats, status = get_latest_stats_and_model()
    if status != "Success": return None, status
    
    fixtures = get_fixtures_api()
    if fixtures.empty: return None, "⚠️ No matches found via API."

    predictions = []
    for _, row in fixtures.iterrows():
        h_s = strict_match(row['Home'], row['League_Code'], stats)
        a_s = strict_match(row['Away'], row['League_Code'], stats)
        
        if h_s is not None and a_s is not None:
            feat = pd.DataFrame([{
                'H_Att': h_s['avg_GF'], 'H_Def': h_s['avg_GA'], 'H_SOT': h_s['avg_SOT'], 'H_Corn': h_s['avg_Corn'],
                'A_Att': a_s['avg_GF'], 'A_Def': a_s['avg_GA'], 'A_SOT': a_s['avg_SOT'], 'A_Corn': a_s['avg_Corn']
            }])
            pred = model.predict(feat)[0]
            predictions.append({
                'Date': row['Date'], 'League': row['League_Code'], 'Match': f"{row['Home']} vs {row['Away']}",
                'Pred': round(pred, 2), 'Tip': "🔥 Over 2.5" if pred > 2.65 else ("Under 2.5" if pred < 2.35 else "Risky")
            })
    return (pd.DataFrame(predictions).sort_values(by='Pred', ascending=False), "Success") if predictions else (None, "❌ No stats matches.")

# ==========================================
# 🖥️ UI LAYOUT
# ==========================================

st.title("⚽ 2-Up Master Suite")
tab1, tab2 = st.tabs(["⚡ Live Predictor", "🔎 Fixture Analyzer"])

with tab1:
    st.markdown("### Decision Tool (In-Play)")
    sel_league = st.selectbox("Select League", LEAGUES_LIVE)
    try:
        m, s = load_live_system(sel_league)
        c1, c2, c3 = st.columns(3)
        min_in = c1.number_input("Goal Minute", 0, 95, 35)
        h_xg = c2.number_input("Home xG", 0.0, 10.0, 1.2)
        a_xg = c3.number_input("Away xG", 0.0, 10.0, 0.5)
        
        c4, c5, c6 = st.columns(3)
        stk, b_o, l_o = c4.number_input("Stake", 10.0), c5.number_input("Bookie Odds", 2.0), c6.number_input("Exchange Odds", 1.1)
        
        if st.button("Analyze Match"):
            txg = h_xg + a_xg
            elf = (90 - min_in) * txg
            inp = pd.DataFrame([[min_in, txg, elf]], columns=['minute_2_0', 'xg_at_event', 'early_lead_factor'])
            prob = m.predict_proba(s.transform(inp))[0][1]
            
            st.metric("Comeback Probability", f"{prob:.1%}")
            if prob >= 0.05: st.success("✅ LET IT RIDE")
            else: st.error("🛑 CASH OUT NOW")
    except Exception as e: st.warning(f"Live Predictor Error: {e}")

with tab2:
    st.markdown("### Upcoming Fixtures (Automated)")
    if st.button("🔄 Fetch & Analyze Games"):
        with st.spinner("Updating stats and running model..."):
            df, msg = run_analysis_pipeline()
            if df is not None:
                st.dataframe(df, hide_index=True, use_container_width=True)
            else: st.warning(msg)
