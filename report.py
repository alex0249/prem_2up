import pandas as pd
import requests
import json
import logging
from bs4 import BeautifulSoup

# Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')

LEAGUES = ['EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1']
BASE_URL = "https://understat.com/league"
CURRENT_YEAR = 2023 # Update this to the current season year (e.g., 2024 when applicable)

def get_league_stats(league):
    url = f"{BASE_URL}/{league}/{CURRENT_YEAR}"
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        scripts = soup.find_all('script')
        
        teams_data = None
        for script in scripts:
            if 'teamsData' in script.text:
                start = script.text.index("('") + 2
                end = script.text.index("')")
                json_str = script.text[start:end].encode('utf8').decode('unicode_escape')
                teams_data = json.loads(json_str)
                break
        
        if not teams_data: return []

        stats_list = []
        for team_id, stats in teams_data.items():
            # Calculate Per 90 Stats
            matches = int(stats['history'][0]['played'])
            if matches < 5: continue # Skip if season just started
            
            xG = sum(h['xG'] for h in stats['history'])
            xGA = sum(h['xGA'] for h in stats['history'])
            
            xG_p90 = xG / matches
            xGA_p90 = xGA / matches
            
            # THE FORMULA: 
            # We want teams that Score (High xG) AND Concede (High xGA).
            # Why Concede? Because bad defenses have higher Odds (2.5+), 
            # which is critical for the 2-Up Strategy profit.
            chaos_score = (xG_p90 * 1.5) + (xGA_p90 * 1.0)
            
            stats_list.append({
                'League': league,
                'Team': stats['title'],
                'xG/90': round(xG_p90, 2),
                'xGA/90': round(xGA_p90, 2),
                'Chaos_Score': round(chaos_score, 2)
            })
            
        return stats_list

    except Exception as e:
        logging.error(f"Error scraping {league}: {e}")
        return []

def generate_report():
    print("🔎 SCANNING EUROPE FOR 'GLASS CANNON' TEAMS...")
    all_teams = []
    
    for league in LEAGUES:
        teams = get_league_stats(league)
        all_teams.extend(teams)
        print(f"  > Analyzed {league}...")

    df = pd.DataFrame(all_teams)
    
    # Sort by Chaos Score
    df = df.sort_values(by='Chaos_Score', ascending=False)
    
    # Filter: We only want teams with decent attacking output
    # (No point betting on a team that concedes 3 but scores 0)
    df = df[df['xG/90'] > 1.3]

    print("\n" + "="*60)
    print("🎯 TOP 20 TEAMS TO WATCH FOR 2-UP OFFERS")
    print("Criteria: High Scoring + Defensive Liability = Volatility")
    print("="*60)
    
    # Clean output
    print(df[['League', 'Team', 'xG/90', 'xGA/90', 'Chaos_Score']].head(20).to_string(index=False))
    
    print("\n" + "="*60)
    print("HOW TO USE THIS LIST:")
    print("1. Look for matches where these teams are playing.")
    print("2. CHECK THE ODDS: Are they between 2.0 and 3.0?")
    print("   - If Yes: This is a PRIME 2-Up Opportunity.")
    print("   - If No (e.g. 1.20 odds): Skip. No value.")
    print("="*60)
    
    # Save to CSV for reference
    df.to_csv("weekend_watchlist.csv", index=False)

if __name__ == "__main__":
    generate_report()