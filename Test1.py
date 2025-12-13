import pandas as pd
import numpy as np
import requests
import json
import time
import random
import os
import joblib
import logging
from bs4 import BeautifulSoup
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# CONFIGURATION
LEAGUES = ['EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1']
YEARS = [2017,2018, 2019, 2020, 2021, 2022, 2023] # 5 Years of history
BASE_URL = "https://understat.com"

class UnderstatScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})

    def get_match_ids(self, league, year):
        """Scrapes the season page to get the list of played matches."""
        url = f"{BASE_URL}/league/{league}/{year}"
        try:
            res = self.session.get(url)
            soup = BeautifulSoup(res.content, 'html.parser')
            scripts = soup.find_all('script')
            
            for script in scripts:
                if 'datesData' in script.text:
                    # Extract JSON
                    start = script.text.index("('") + 2
                    end = script.text.index("')")
                    json_str = script.text[start:end].encode('utf8').decode('unicode_escape')
                    data = json.loads(json_str)
                    # Return only played matches (isResult = True)
                    return [x['id'] for x in data if x['isResult'] == True]
        except Exception as e:
            logging.error(f"Failed to get season data for {league} {year}: {e}")
            return []

    def get_match_data(self, match_id):
        """Scrapes a single match to find '2-Up' scenarios."""
        url = f"{BASE_URL}/match/{match_id}"
        try:
            # RATE LIMITING: Random sleep to look like a human
            time.sleep(random.uniform(0.8, 1.5))
            
            res = self.session.get(url)
            soup = BeautifulSoup(res.content, 'html.parser')
            scripts = soup.find_all('script')
            
            shots_data = None
            for script in scripts:
                if 'shotsData' in script.text:
                    start = script.text.index("('") + 2
                    end = script.text.index("')")
                    json_str = script.text[start:end].encode('utf8').decode('unicode_escape')
                    shots_data = json.loads(json_str)
                    break
            
            if not shots_data: return None

            return self._analyze_timeline(shots_data)
        except Exception as e:
            logging.error(f"Error scraping match {match_id}: {e}")
            return None

    def _analyze_timeline(self, shots):
        # Reconstruct the match timeline
        timeline = []
        for side in ['h', 'a']:
            for s in shots[side]:
                timeline.append({
                    'min': int(s['minute']),
                    'type': 'goal' if s['result'] == 'Goal' else 'shot',
                    'side': side,
                    'xg': float(s['xG'])
                })
        timeline.sort(key=lambda x: x['min'])

        h_score, a_score = 0, 0
        went_2up = False
        data_point = None
        
        # Walk through the match
        for event in timeline:
            if event['type'] == 'goal':
                if event['side'] == 'h': h_score += 1
                else: a_score += 1
                
                # Check for 2-0 or 0-2
                if not went_2up:
                    if (h_score == 2 and a_score == 0) or (a_score == 2 and h_score == 0):
                        went_2up = True
                        leader = 'h' if h_score == 2 else 'a'
                        
                        # Calculate total xG at this exact moment
                        # (Sum of all xG from start of match until now)
                        xg_sum = sum(e['xg'] for e in timeline if e['min'] <= event['min'])
                        
                        data_point = {
                            'minute_2_0': event['min'],
                            'xg_at_event': xg_sum,
                            'leader': leader,
                            'final_h': 0, # Placeholders, updated later
                            'final_a': 0
                        }

        if not went_2up or not data_point:
            return None

        # Determine if Comeback happened (Draw or Loss for leader)
        comeback = 0
        if data_point['leader'] == 'h':
            if h_score <= a_score: comeback = 1
        else:
            if a_score <= h_score: comeback = 1
            
        return {
            'minute_2_0': data_point['minute_2_0'],
            'xg_at_event': data_point['xg_at_event'],
            'target_comeback': comeback
        }

def run_pipeline():
    scraper = UnderstatScraper()
    
    for league in LEAGUES:
        logging.info(f"STARTING PROCESS FOR LEAGUE: {league}")
        league_data = []
        filename = f"data_{league}.csv"
        
        # 1. SCRAPE (or load if already done)
        if os.path.exists(filename):
            logging.info("  Found existing CSV. Loading data...")
            df = pd.read_csv(filename)
        else:
            for year in YEARS:
                logging.info(f"  Scraping {year}...")
                ids = scraper.get_match_ids(league, year)
                logging.info(f"    Found {len(ids)} matches. Scanning for 2-Ups...")
                
                for i, mid in enumerate(ids):
                    result = scraper.get_match_data(mid)
                    if result:
                        league_data.append(result)
                    
                    if i % 50 == 0 and i > 0:
                        logging.info(f"    Processed {i}/{len(ids)} matches...")
            
            df = pd.DataFrame(league_data)
            if not df.empty:
                df.to_csv(filename, index=False)
                logging.info(f"  Saved {len(df)} training examples to {filename}")
            else:
                logging.warning(f"  No 2-Ups found for {league} (Check scraper logic or connection).")
                continue

        # 2. TRAIN
        if df.empty: continue
        
        logging.info(f"  Training Model for {league}...")
        
        # Feature Engineering
        df['early_lead_factor'] = (90 - df['minute_2_0']) * df['xg_at_event']
        X = df[['minute_2_0', 'xg_at_event', 'early_lead_factor']]
        y = df['target_comeback']
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # SVM Training
        svm = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
        svm.fit(X_scaled, y)
        
        # 3. SAVE ARTIFACTS
        joblib.dump(svm, f"model_{league}.pkl")
        joblib.dump(scaler, f"scaler_{league}.pkl")
        logging.info(f"  SUCCESS: Saved model_{league}.pkl")

    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("You now have 5 real models ready for the App.")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()