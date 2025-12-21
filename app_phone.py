import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import joblib
import os
import warnings
import tempfile
import datetime
import time
import betfairlightweight
from betfairlightweight import filters
from sklearn.linear_model import Ridge
from fuzzywuzzy import process
from scipy.stats import poisson

warnings.filterwarnings('ignore')

# ==========================================
# 🔑 CONFIGURATION
# ==========================================

# 1. API KEYS (FOOTBALL DATA)
FD_API_KEY = 'c46720c50cc04c499506a757b9a00259' 
BASE_URL_API = 'https://api.football-data.org/v4/matches'
LEAGUE_CODES_API = ['PL', 'PD', 'BL1', 'SA', 'FL1']

# Training Data Config
SEASONS_TO_FETCH = ['2425', '2526']
BASE_URL_DATA = "https://www.football-data.co.uk/mmz4281"
LEAGUE_MAP = {'E0': 'PL', 'SP1': 'PD', 'D1': 'BL1', 'I1': 'SA', 'F1': 'FL1'}

# ==========================================
# 🔐 BETFAIR AUTHENTICATION HELPER (FIXED)
# ==========================================

def get_betfair_client():
    """
    Reconstructs certificates from Streamlit Secrets into a temp directory
    and logs in to Betfair.
    """
    if 'betfair' not in st.secrets:
        return None

    secrets = st.secrets["betfair"]
    
    # 1. Create a temporary directory
    # betfairlightweight requires a folder path, not individual file paths.
    temp_dir = tempfile.mkdtemp()
    
    # 2. Write the certificate files with the EXACT names required by the library
    crt_path = os.path.join(temp_dir, "client-2048.crt")
    key_path = os.path.join(temp_dir, "client-2048.key")
    
    with open(crt_path, "w") as f:
        f.write(secrets["cert_file"])
        
    with open(key_path, "w") as f:
        f.write(secrets["key_file"])

    try:
        # 3. Pass the DIRECTORY (temp_dir) to the client
        trading = betfairlightweight.APIClient(
            username=secrets["username"],
            password=secrets["password"],
            app_key=secrets["app_key"],
            certs=temp_dir,
            locale='en'
        )
        trading.login()
        return trading
    except Exception as e:
        st.error(f"❌ Betfair Login Failed: {e}")
        return None

# ==========================================
# ⚙️ AUTOMATED BACKEND: DATA & TRAINING
# ==========================================

@st.cache_data(ttl=86400)
def get_latest_stats_and_model():
    """Trains Home/Away Ridge Models"""
    all_data = []
    
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
        return None, None, None, "❌ Failed to download historical data."

    full_df = pd.concat(all_data, ignore_index=True)
    
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
    
    train_df = df.merge(team_df[['Date', 'Team', 'avg_GF', 'avg_GA', 'avg_SOT', 'avg_Corn']], 
                      left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='inner')
    train_df = train_df.rename(columns={'avg_GF': 'H_Att', 'avg_GA': 'H_Def', 'avg_SOT': 'H_SOT', 'avg_Corn': 'H_Corn'}).drop(columns=['Team'])
    
    train_df = train_df.merge(team_df[['Date', 'Team', 'avg_GF', 'avg_GA', 'avg_SOT', 'avg_Corn']], 
                      left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='inner')
    train_df = train_df.rename(columns={'avg_GF': 'A_Att', 'avg_GA': 'A_Def', 'avg_SOT': 'A_SOT', 'avg_Corn': 'A_Corn'}).drop(columns=['Team'])
    
    train_df = train_df.dropna()

    features = ['H_Att', 'H_Def', 'H_SOT', 'H_Corn', 'A_Att', 'A_Def', 'A_SOT', 'A_Corn']
    
    model_home = Ridge(alpha=10)
    model_home.fit(train_df[features], train_df['FTHG'])
    
    model_away = Ridge(alpha=10)
    model_away.fit(train_df[features], train_df['FTAG'])
    
    return model_home, model_away, latest_stats, "Success"

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def simulate_match(xg_h, xg_a):
    max_goals = 10
    prob_h = poisson.pmf(np.arange(max_goals), xg_h)
    prob_a = poisson.pmf(np.arange(max_goals), xg_a)
    matrix = np.outer(prob_h, prob_a)
    
    indices = np.indices((max_goals, max_goals))
    mask_o25 = (indices[0] + indices[1]) > 2.5
    prob_o25 = np.sum(matrix[mask_o25])
    
    prob_home_win = np.sum(matrix[indices[0] > indices[1]])
    prob_away_win = np.sum(matrix[indices[0] < indices[1]])
    
    return prob_o25, prob_home_win, prob_away_win

def strict_match(team_name, league_code, stats_db):
    league_teams = stats_db[stats_db['League_ID'] == league_code]
    if league_teams.empty: return None
    choices = league_teams['Team'].unique()
    match, score = process.extractOne(team_name, choices)
    return league_teams[league_teams['Team'] == match].iloc[0] if score > 65 else None

def get_fixtures_api():
    headers = {'X-Auth-Token': FD_API_KEY}
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

def fetch_betfair_odds(client, home_team, away_team):
    """
    Searches Betfair for the match and returns market prices.
    Returns: (Home Odds, Away Odds)
    """
    if not client: return None, None
    
    search_query = f"{home_team} {away_team}"
    
    event_filter = filters.market_filter(
        text_query=search_query,
        event_type_ids=['1'],
        market_type_codes=['MATCH_ODDS']
    )
    
    try:
        markets = client.betting.list_market_catalogue(
            filter=event_filter,
            max_results=1,
            market_projection=['RUNNER_METADATA', 'MARKET_START_TIME']
        )
    except: return None, None

    if not markets: return None, None

    market = markets[0]
    market_id = market.market_id
    
    selection_map = {} 
    for runner in market.runners:
        r_name = runner.runner_name
        s_home = process.extractOne(r_name, [home_team])[1]
        s_away = process.extractOne(r_name, [away_team])[1]
        
        if s_home > s_away and s_home > 60:
            selection_map[runner.selection_id] = 'Home'
        elif s_away > s_home and s_away > 60:
            selection_map[runner.selection_id] = 'Away'

    price_filter = filters.price_projection(price_data=['EX_BEST_OFFERS'])
    
    try:
        books = client.betting.list_market_book(
            market_ids=[market_id],
            price_projection=price_filter
        )
    except: return None, None

    odds_home = None
    odds_away = None

    for book in books:
        for runner in book.runners:
            if runner.selection_id in selection_map:
                if runner.ex.available_to_back:
                    best_price = runner.ex.available_to_back[0].price
                    if selection_map[runner.selection_id] == 'Home':
                        odds_home = best_price
                    elif selection_map[runner.selection_id] == 'Away':
                        odds_away = best_price
    
    return odds_home, odds_away

# ==========================================
# 🚀 MAIN PIPELINE
# ==========================================

def run_analysis_pipeline():
    model_h, model_a, stats, status = get_latest_stats_and_model()
    if status != "Success": return None, status
    
    fixtures = get_fixtures_api()
    if fixtures.empty: return None, "⚠️ No matches found via API."

    bf_client = get_betfair_client()
    if not bf_client:
        st.warning("⚠️ Could not log in to Betfair. Live EV will be missing (Check Secrets).")

    predictions = []
    
    progress_bar = st.progress(0)
    total_games = len(fixtures)
    
    for i, row in fixtures.iterrows():
        progress_bar.progress((i + 1) / total_games)
        
        h_s = strict_match(row['Home'], row['League_Code'], stats)
        a_s = strict_match(row['Away'], row['League_Code'], stats)
        
        if h_s is not None and a_s is not None:
            feat = pd.DataFrame([{
                'H_Att': h_s['avg_GF'], 'H_Def': h_s['avg_GA'], 'H_SOT': h_s['avg_SOT'], 'H_Corn': h_s['avg_Corn'],
                'A_Att': a_s['avg_GF'], 'A_Def': a_s['avg_GA'], 'A_SOT': a_s['avg_SOT'], 'A_Corn': a_s['avg_Corn']
            }])
            
            xg_h = max(0.1, model_h.predict(feat)[0])
            xg_a = max(0.1, model_a.predict(feat)[0])
            
            prob_o25, prob_h_win, prob_a_win = simulate_match(xg_h, xg_a)
            
            bf_home, bf_away = None, None
            if bf_client:
                time.sleep(0.2)
                bf_home, bf_away = fetch_betfair_odds(bf_client, row['Home'], row['Away'])

            if prob_h_win > prob_a_win:
                pick = row['Home']
                prob_win = prob_h_win
                live_odds = bf_home
            else:
                pick = row['Away']
                prob_win = prob_a_win
                live_odds = bf_away

            ev_str = "N/A"
            if live_odds:
                ev = (prob_win * live_odds) - 1
                ev_str = f"{ev:.2f}"
                if ev > 0: ev_str = f"✅ {ev_str}"
            
            predictions.append({
                'Match': f"{row['Home']} vs {row['Away']}",
                'xG': f"{xg_h:.1f}-{xg_a:.1f}",
                '2Up Pick': pick,
                'Win Prob': f"{prob_win:.1%}",
                'Fair Odds': round(1/prob_win, 2),
                'BF Odds': live_odds if live_odds else "Missing",
                'EV': ev_str
            })
            
    return pd.DataFrame(predictions), "Success"

# ==========================================
# 🖥️ UI LAYOUT
# ==========================================

st.title("⚽ 2-Up Master Suite (Betfair Integrated)")

if st.button("🔄 Analyze Expected Value (With Betfair Odds)"):
    with st.spinner("Training models & Scanning Betfair Markets..."):
        df, msg = run_analysis_pipeline()
        if df is not None:
            st.dataframe(
                df, 
                # FIX: use_container_width=True is the safest option. 
                # If you get a warning, you can ignore it, or change it to width="stretch"
                use_container_width=True, 
                column_config={
                    "Fair Odds": st.column_config.NumberColumn(
                        "Fair Odds",
                        help="The minimum odds required for this to be a break-even bet.",
                        format="%.2f"
                    ),
                    "BF Odds": st.column_config.NumberColumn(
                        "Exchange Odds",
                        help="Live available back odds from Betfair.",
                        format="%.2f"
                    )
                }
            )
        else:
            st.warning(msg)


