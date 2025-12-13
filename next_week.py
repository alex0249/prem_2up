import requests
import pandas as pd
import pickle
import datetime
from fuzzywuzzy import process
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 🔑 CONFIGURATION
# ==========================================
API_KEY = 'c46720c50cc04c499506a757b9a00259' 

LEAGUE_CODES = ['PL', 'PD', 'BL1', 'SA', 'FL1']
BASE_URL = 'https://api.football-data.org/v4/matches'

# ==========================================
# 1. FETCH FIXTURES
# ==========================================
def get_fixtures_api():
    if 'YOUR_API' in API_KEY:
        print("❌ Error: Please insert your API Key.")
        return pd.DataFrame()

    print("🌍 Fetching fixtures from API...")
    headers = {'X-Auth-Token': API_KEY}
    
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    next_week = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    
    params = {
        'dateFrom': today, 'dateTo': next_week,
        'competitions': ','.join(LEAGUE_CODES),
        'status': 'SCHEDULED'
    }
    
    try:
        r = requests.get(BASE_URL, headers=headers, params=params)
        data = r.json()
        
        fixtures = []
        for m in data.get('matches', []):
            fixtures.append({
                'Date': m['utcDate'][:10],
                'League_Code': m['competition']['code'], # e.g., 'PL', 'BL1'
                'League_Name': m['competition']['name'],
                'Home': m['homeTeam']['name'],
                'Away': m['awayTeam']['name']
            })
        return pd.DataFrame(fixtures)
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        return pd.DataFrame()

# ==========================================
# 2. STRICT MATCHING LOGIC
# ==========================================
def strict_match(team_name, league_code, stats_db):
    # 1. FILTER: Only look at teams that belong to this league
    # This prevents matching "Arsenal" (PL) with "Koln" (BL1)
    league_teams_df = stats_db[stats_db['League_ID'] == league_code]
    
    if league_teams_df.empty:
        return None
        
    choices = league_teams_df['Team'].unique()
    
    # 2. FUZZY MATCH: Now find the name in this smaller list
    match, score = process.extractOne(team_name, choices)
    
    if score > 65:
        # Return the stats for that specific team
        return league_teams_df[league_teams_df['Team'] == match].iloc[0]
        
    return None

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    # Load
    try:
        with open('soccer_model.pkl', 'rb') as f: model = pickle.load(f)
        stats = pd.read_csv('latest_team_stats.csv')
    except:
        print("❌ Run 'train_model.py' first!"); exit()

    # Get Games
    fixtures = get_fixtures_api()
    
    if fixtures.empty:
        print("⚠️ No games found via API.")
    else:
        print(f"✅ Found {len(fixtures)} matches. Analyzing...")
        print("-" * 60)
        
        predictions = []
        for _, row in fixtures.iterrows():
            # Pass the League Code (e.g., 'PL') to the matcher
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

        if predictions:
            df_res = pd.DataFrame(predictions).sort_values(by='Pred', ascending=False)
            print(df_res[['Date', 'League', 'Match', 'Pred', 'Tip']].to_string(index=False))
        else:
            print("❌ No valid matches found (Teams missing from stats DB).")