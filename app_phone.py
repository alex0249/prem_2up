import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import json
from bs4 import BeautifulSoup
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="2-Up Master Suite", layout="centered")

# Defines which year to scrape for the Scout (Update this in August each year)
# Since today is Dec 2025, we are in the 2025/2026 season.
CURRENT_SEASON_YEAR = 2025 

LEAGUES = ['EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1']

# ==========================================
# 1. LIVE PREDICTOR LOGIC
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
# 2. SCOUTING LOGIC (The New Feature)
# ==========================================

@st.cache_data(ttl=3600) # Cache data for 1 hour to prevent constant re-scraping
def get_scouting_report():
    base_url = "https://understat.com/league"
    all_teams = []
    
    progress_text = "Scanning Europe's Top Leagues..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, league in enumerate(LEAGUES):
        url = f"{base_url}/{league}/{CURRENT_SEASON_YEAR}"
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
            
            if teams_data:
                for team_id, stats in teams_data.items():
                    matches = int(stats['history'][0]['played'])
                    if matches < 5: continue 
                    
                    xG = sum(h['xG'] for h in stats['history'])
                    xGA = sum(h['xGA'] for h in stats['history'])
                    
                    xG_p90 = xG / matches
                    xGA_p90 = xGA / matches
                    
                    # Chaos Score Formula
                    chaos_score = (xG_p90 * 1.5) + (xGA_p90 * 1.0)
                    
                    all_teams.append({
                        'League': league,
                        'Team': stats['title'],
                        'xG_For': round(xG_p90, 2),
                        'xG_Against': round(xGA_p90, 2),
                        'Chaos_Score': round(chaos_score, 2)
                    })
        except Exception as e:
            st.error(f"Error scraping {league}: {e}")
            
        # Update Progress Bar
        my_bar.progress((i + 1) / len(LEAGUES), text=f"Finished analyzing {league}...")
            
    my_bar.empty()
    return pd.DataFrame(all_teams)

# ==========================================
# 3. UI LAYOUT
# ==========================================

st.title("⚽ 2-Up Master Suite")

# Create Tabs
tab1, tab2 = st.tabs(["⚡ Live Predictor", "🔎 Weekend Scout"])

# --- TAB 1: LIVE PREDICTOR ---
with tab1:
    st.markdown("### Decision Tool (In-Play)")
    
    selected_league = st.selectbox("Select League", LEAGUES, key="league_select")

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
            rec_color = "green"
        else:
            st.error("🛑 DECISION: CASH OUT NOW (Lead is Safe)")
            rec_color = "red"
            
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

# --- TAB 2: WEEKEND SCOUT ---
with tab2:
    st.markdown("### Volatility Scanner")
    st.info("Use this on Fridays to find the best teams to bet on. Look for high Chaos Scores with Odds > 2.0.")
    
    if st.button("🔄 Scan Top 5 Leagues"):
        df_scout = get_scouting_report()
        
        # Sort and Filter
        df_scout = df_scout.sort_values(by='Chaos_Score', ascending=False)
        df_scout = df_scout[df_scout['xG_For'] > 1.2] # Filter out boring teams
        
        st.success("Scan Complete! Here are the 'Glass Cannon' teams.")
        
        # Display as a clean interactive table
        st.dataframe(
            df_scout,
            column_config={
                "Chaos_Score": st.column_config.ProgressColumn(
                    "Volatility Rating",
                    help="Higher score = More likely to score AND concede",
                    format="%.2f",
                    min_value=0,
                    max_value=5,
                ),
                "xG_For": st.column_config.NumberColumn("xG Scored", format="%.2f"),
                "xG_Against": st.column_config.NumberColumn("xG Conceded", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("#### 💡 Strategy")
        st.markdown("""
        1.  Pick teams from the top of this list (High Volatility).
        2.  Check Oddschecker: **Are their odds between 2.0 and 3.0?**
        3.  If **YES**: This is a prime 2-Up Candidate.
        4.  If **NO** (Odds < 1.4): Skip. No value in the trade.
        """)
