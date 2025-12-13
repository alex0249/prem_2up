import pandas as pd
import numpy as np
import requests
import io
import pickle
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
SEASONS_TO_FETCH = ['2425', '2526'] 
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Map CSV file codes to the API codes we will use later
LEAGUE_MAPPING = {
    'E0': 'PL',   # Premier League
    'SP1': 'PD',  # La Liga
    'D1': 'BL1',  # Bundesliga
    'I1': 'SA',   # Serie A
    'F1': 'FL1'   # Ligue 1
}

def get_training_data():
    print("📥 Downloading historical stats...")
    all_data = []
    
    for s in SEASONS_TO_FETCH:
        for csv_code, api_code in LEAGUE_MAPPING.items():
            url = f"{BASE_URL}/{s}/{csv_code}.csv"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    df = pd.read_csv(io.StringIO(response.content.decode('latin1')))
                    if 'Date' in df.columns and 'HomeTeam' in df.columns:
                        # CRITICAL: Tag the league so we don't mix them up later
                        df['League_ID'] = api_code 
                        all_data.append(df)
            except Exception:
                continue

    # Fallback to older data if current season is empty
    if not all_data:
        print("⚠️ Warning: Fetching older seasons (23/24)...")
        for csv_code, api_code in LEAGUE_MAPPING.items():
            url = f"{BASE_URL}/2324/{csv_code}.csv"
            try:
                df = pd.read_csv(url, encoding='latin1')
                df['League_ID'] = api_code
                all_data.append(df)
            except: pass

    if not all_data:
        raise ValueError("❌ Failed to download any data.")

    full_df = pd.concat(all_data, ignore_index=True)
    return process_features(full_df)

def process_features(df):
    print("⚙️ Processing features...")
    
    # 1. Cleanup
    req_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HST', 'AST', 'HC', 'AC', 'League_ID']
    df = df[df.columns.intersection(req_cols)].dropna()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date')
    
    # 2. Reshape for Rolling Stats
    # We must keep League_ID in the team stats
    home_stats = df[['Date', 'HomeTeam', 'League_ID', 'FTHG', 'FTAG', 'HST', 'HC']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GF', 'FTAG': 'GA', 'HST': 'SOT', 'HC': 'Corn'}
    )
    away_stats = df[['Date', 'AwayTeam', 'League_ID', 'FTAG', 'FTHG', 'AST', 'AC']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GF', 'FTHG': 'GA', 'AST': 'SOT', 'AC': 'Corn'}
    )
    
    team_df = pd.concat([home_stats, away_stats]).sort_values('Date')
    
    # 3. Rolling Averages
    cols = ['GF', 'GA', 'SOT', 'Corn']
    for c in cols:
        team_df[f'avg_{c}'] = team_df.groupby('Team')[c].transform(
            lambda x: x.rolling(5, closed='left').mean()
        )
    
    # 4. Save Latest Stats (With League_ID included!)
    latest_stats = team_df.groupby('Team').last().reset_index()
    
    # 5. Merge for Training
    df = df.merge(team_df[['Date', 'Team', 'avg_GF', 'avg_GA', 'avg_SOT', 'avg_Corn']], 
                  left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='inner')
    df = df.rename(columns={'avg_GF': 'H_Att', 'avg_GA': 'H_Def', 'avg_SOT': 'H_SOT', 'avg_Corn': 'H_Corn'}).drop(columns=['Team'])
    
    df = df.merge(team_df[['Date', 'Team', 'avg_GF', 'avg_GA', 'avg_SOT', 'avg_Corn']], 
                  left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='inner')
    df = df.rename(columns={'avg_GF': 'A_Att', 'avg_GA': 'A_Def', 'avg_SOT': 'A_SOT', 'avg_Corn': 'A_Corn'}).drop(columns=['Team'])
    
    df['total_goals'] = df['FTHG'] + df['FTAG']
    return df.dropna(), latest_stats

if __name__ == "__main__":
    train_df, latest_stats = get_training_data()
    
    features = ['H_Att', 'H_Def', 'H_SOT', 'H_Corn', 'A_Att', 'A_Def', 'A_SOT', 'A_Corn']
    target = 'total_goals'
    
    print(f"🧠 Training Ridge Model on {len(train_df)} matches...")
    model = Ridge(alpha=10)
    model.fit(train_df[features], train_df[target])
    
    # SAVE
    with open('soccer_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    latest_stats.to_csv('latest_team_stats.csv', index=False)
    print("✅ Success! 'latest_team_stats.csv' now contains League IDs.")