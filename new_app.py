import streamlit as st
import pandas as pd
import pickle
import datetime
import requests
import joblib
import os
import warnings
from fuzzywuzzy import process

warnings.filterwarnings('ignore')

# ==========================================
# 🔑 CONFIGURATION
# ==========================================
st.set_page_config(page_title="2-Up Master Suite", layout="centered")

# API Configuration (For Tab 2)
API_KEY = 'c46720c50cc04c499506a757b9a00259' # Security Warning: Use Environment Variables in production
BASE_URL = 'https://api.football-data.org/v4/matches'
LEAGUE_CODES_API = ['PL', 'PD', 'BL1', 'SA', 'FL1'] # API specific codes

# Live Predictor Configuration (For Tab 1)
LEAGUES_LIVE = ['EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1']

# ==========================================
# 1. HELPER FUNCTIONS: LIVE PREDICTOR (TAB 1)
# ==========================================

def load_system(league_name):
    """Loads the specific brain for the selected league."""
    model_file = f"model_{league_name}.pkl"
    scaler_file = f"scaler_{league_name}.pkl"

    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        raise FileNotFoundError(f"⚠️ Files for {league_name} not found! Upload '{model_file}' and '{scaler_file}'.")

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    return model, scaler

def calculate_probability(minute, home_xg, away_xg, model, scaler):
    total_xg = home_xg + away_xg
    early_lead_factor = (90 - minute) * total_xg
    
    input_data = pd.DataFrame([[minute, total_xg, early_lead_factor]], 
                              columns=['minute_2_0', 'xg_at_event', 'early_lead_factor'])
    
    input_scaled = scaler.transform(input_data)
    probs = model.predict_proba(input_scaled)
    return probs[0][1]

def calculate_financials(stake, back_odds, lay_odds_current):
    bookie_profit = stake * (back_odds - 1)
    cost_to_exit = stake * (lay_odds_current - 1) 
    guaranteed_profit = bookie_profit - cost_to_exit
    potential_upside = bookie_profit + stake 
    return guaranteed_profit, potential_upside

# ==========================================
# 2. HELPER FUNCTIONS: FIXTURE ANALYZER (TAB 2)
# ==========================================

def get_fixtures_api():
    if 'YOUR_API' in API_KEY:
        st.error("❌ Error: Please insert your API Key.")
        return pd.DataFrame()

    headers = {'X-Auth-Token': API_KEY}
    
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    next_week = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    
    params = {
        'dateFrom': today, 'dateTo': next_week,
        'competitions': ','.join(LEAGUE_CODES_API),
        'status': 'SCHEDULED'
    }
    
    try:
        r = requests.get(BASE_URL, headers=headers, params=params)
        data = r.json()
        
        fixtures = []
        matches = data.get('matches', [])
        
        if not matches:
            return pd.DataFrame()

        for m in matches:
            fixtures.append({
                'Date': m['utcDate'][:10],
                'League_Code': m['competition']['code'],
                'League_Name': m['competition']['name'],
                'Home': m['homeTeam']['name'],
                'Away': m['awayTeam']['name']
            })
        return pd.DataFrame(fixtures)
    except Exception as e:
        st.error(f"⚠️ API Error: {e}")
        return pd.DataFrame()

def strict_match(team_name, league_code, stats_db):
    # Filter by league first to avoid cross-league confusion
    league_teams_df = stats_db[stats_db['League_ID'] == league_code]
    
    if league_teams_df.empty:
        return None
        
    choices = league_teams_df['Team'].unique()
    
    # Fuzzy match
    match, score = process.extractOne(team_name, choices)
    
    if score > 65:
        return league_teams_df[league_teams_df['Team'] == match].iloc[0]
        
    return None

@st.cache_data(ttl=3600)
def run_analysis_pipeline():
    # 1. Load Model & Stats
    try:
        with open('soccer_model.pkl', 'rb') as f: 
            model = pickle.load(f)
        stats = pd.read_csv('latest_team_stats.csv')
    except Exception as e:
        return None, f"❌ Missing Files: {e}. Ensure 'soccer_model.pkl' and 'latest_team_stats.csv' are in the directory."

    # 2. Get Games
    fixtures = get_fixtures_api()
    
    if fixtures.empty:
        return None, "⚠️ No matches found via API or API limit reached."

    # 3. Predict
    predictions = []
    for _, row in fixtures.iterrows():
        h_stats = strict_match(row['Home'], row['League_Code'], stats)
        a_stats = strict_match(row['Away'], row['League_Code'], stats)
        
        if h_stats is not None and a_stats is not None:
            features = pd.DataFrame([{
                'H_Att': h_stats['avg_GF'], 'H_Def': h_stats['avg_GA'],
                'H_SOT': h_stats['avg_SOT'], 'H_Corn': h_stats['avg_Corn'],
                'A_Att': a_stats['avg_GF'], 'A_Def': a_stats['avg_GA'],
                'A_SOT': a_stats['avg_SOT'], 'A_Corn': a_stats['avg_Corn']
            }])
            
            pred = model.predict(features)[0]
            
            predictions.append({
                'Date': row['Date'],
                'League': row['League_Code'],
                'Match': f"{row['Home']} vs {row['Away']}",
                'Pred': round(pred, 2),
                'Tip': "🔥 Over 2.5" if pred > 2.65 else ("Under 2.5" if pred < 2.35 else "Risky")
            })

    if not predictions:
        return None, "❌ Fixtures found, but no matching stats in database."
        
    df_res = pd.DataFrame(predictions).sort_values(by='Pred', ascending=False)
    return df_res, "Success"

# ==========================================
# 3. UI LAYOUT
# ==========================================

st.title("⚽ 2-Up Master Suite")

# Create Tabs
tab1, tab2 = st.tabs(["⚡ Live Predictor", "🔎 Fixture Analyzer"])

# --- TAB 1: LIVE PREDICTOR ---
with tab1:
    st.markdown("### Decision Tool (In-Play)")
    
    selected_league = st.selectbox("Select League", LEAGUES_LIVE, key="league_select")

    # Load Model
    try:
        model, scaler = load_system(selected_league)
    except Exception as e:
        st.error(f"{e}")
        st.stop()

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1: minute = st.number_input("Goal Minute", 0, 95, 35)
        with col2: h_xg = st.number_input("Home xG", 0.0, 10.0, 1.2, 0.1)
        with col3: a_xg = st.number_input("Away xG", 0.0, 10.0, 0.5, 0.1)

        col4, col5, col6 = st.columns(3)
        with col4: stake = st.number_input("Back Stake (£)", value=50.0)
        with col5: back_odds = st.number_input("Bookie Odds", value=2.50)
        with col6: current_odds = st.number_input("Current Exchange Odds", value=1.10)

    if st.button("Analyze Match", type="primary"):
        prob = calculate_probability(minute, h_xg, a_xg, model, scaler)
        cash_out_profit, max_upside = calculate_financials(stake, back_odds, current_odds)
        
        st.divider()
        
        # AI Verdict
        THRESHOLD_BET = 0.05
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Comeback Probability", f"{prob:.1%}")
        
        if prob >= THRESHOLD_BET:
            st.success("✅ DECISION: LET IT RIDE (Risk/Reward Favorable)")
        else:
            st.error("🛑 DECISION: CASH OUT NOW (Lead is Safe)")
            
        # Financials
        st.subheader("Financial Options")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Option A: Cash Out**")
            st.markdown(f"**£{cash_out_profit:.2f}** profit")
            if prob < THRESHOLD_BET: st.caption("👈 Recommended")
        with c2:
            st.markdown(f"**Option B: Wait**")
            st.markdown(f"**£{max_upside:.2f}** potential")
            if prob >= THRESHOLD_BET: st.caption("👈 Recommended")

# --- TAB 2: FIXTURE ANALYZER ---
with tab2:
    st.markdown("### Upcoming Fixtures (API)")
    st.info("Fetches upcoming matches and applies the 'soccer_model.pkl' prediction logic.")
    
    if st.button("🔄 Fetch & Analyze Games"):
        with st.spinner("Fetching data from API and matching stats..."):
            df_results, status_msg = run_analysis_pipeline()
            
        if df_results is not None:
            st.success(f"Analysis Complete. Found {len(df_results)} matches.")
            
            st.dataframe(
                df_results,
                column_config={
                    "Pred": st.column_config.NumberColumn(
                        "Exp. Goals",
                        format="%.2f",
                    ),
                    "Tip": st.column_config.TextColumn(
                        "AI Recommendation",
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning(status_msg)