import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. SYSTEM SETUP & MODEL HANDLING
# ==========================================

MODEL_FILE = 'final_2up_model.pkl'
SCALER_FILE = 'final_2up_scaler.pkl'

def ensure_model_exists():
    """
    Checks if the ML model exists. If not, trains a lightweight 
    placeholder model so the App doesn't crash on first run.
    """
    if not os.path.exists(MODEL_FILE):
        st.warning("⚠️ Main model not found. Generating a temporary model for demonstration...")
        
        # Generate synthetic training data (mimicking the 2015-2023 distribution)
        np.random.seed(42)
        n_samples = 1000
        # Features: Minute (10-90), xG (0.5-5.0), Early_Lead_Factor
        minutes = np.random.randint(10, 90, n_samples)
        xgs = np.random.uniform(0.5, 4.0, n_samples)
        early_lead = (90 - minutes) * xgs
        
        X = pd.DataFrame({
            'minute_2_0': minutes,
            'xg_at_event': xgs,
            'early_lead_factor': early_lead
        })
        
        # Synthetic Target: Earlier leads + high xG diff = Less likely to comeback
        prob = 0.05 + (early_lead * 0.0005) # Dummy logic
        prob = np.clip(prob, 0.0, 1.0)
        y = (np.random.random(n_samples) < prob).astype(int)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=42)
        clf.fit(X_scaled, y)
        
        joblib.dump(clf, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        st.success("Temporary model created. Replace with real data later.")

def load_system():
    ensure_model_exists()
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler

# ==========================================
# 2. CALCULATION ENGINES
# ==========================================

def calculate_probability(minute, home_xg, away_xg, model, scaler):
    total_xg = home_xg + away_xg
    early_lead_factor = (90 - minute) * total_xg
    
    # Prepare input vector matching training data format
    input_data = pd.DataFrame([[minute, total_xg, early_lead_factor]], 
                              columns=['minute_2_0', 'xg_at_event', 'early_lead_factor'])
    
    input_scaled = scaler.transform(input_data)
    probs = model.predict_proba(input_scaled)
    return probs[0][1] # Probability of Class 1 (Comeback)

def calculate_financials(stake, back_odds, lay_odds_current, comm):
    """
    Calculates the financial outcome of Cashing Out vs Letting it Ride.
    """
    # 1. Money already secured from Bookie (2-Up Payout)
    bookie_profit = stake * (back_odds - 1)
    
    # 2. Liability on Exchange (What we stand to lose if 2-0 team wins)
    # Original Lay Liability = Stake * (Lay_Odds_Original - 1) 
    # But we calculate "Exit Cost" based on CURRENT market price.
    
    # To exit a Lay, we must BACK the team at current odds.
    # Cost to exit = Stake / Current_Lay_Odds (Simplified Hedge calc)
    # Note: Exchange formulas vary slightly, this is the 'Cash Out' approximation.
    
    # Standard Cash Out Value on Exchange (if team is winning 2-0, this is negative)
    # If we exit now, we pay a small amount to close the bet because odds represent they are likely to win.
    cost_to_exit = stake * (lay_odds_current - 1) # Assuming we matched stake
    
    # OPTION A: CASH OUT NOW
    # We keep Bookie Profit, but we pay the exchange to close the position.
    # Since odds are low (e.g. 1.05), cost is low.
    guaranteed_profit = bookie_profit - cost_to_exit
    
    # OPTION B: LET IT RIDE (Full Turnaround)
    # If comeback happens: We keep Bookie Profit AND we win the Exchange Liability.
    # (Technically we don't pay out the liability).
    potential_upside = bookie_profit + stake # We win the stake back on exchange
    
    return guaranteed_profit, potential_upside

# ==========================================
# 3. STREAMLIT UI LAYOUT
# ==========================================

st.set_page_config(page_title="2-Up Master Suite", layout="centered")

st.title("⚽ Premier League 2-Up Predictor")
st.markdown("### Hybrid Strategy: AI Prediction + Exit Calculator")

# Load Model
try:
    model, scaler = load_system()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- INPUT SECTION ---
with st.container():
    st.markdown("#### 1. Match Situation (Live Data)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        minute = st.number_input("Goal Minute", min_value=0, max_value=95, value=35)
    with col2:
        h_xg = st.number_input("Home xG", min_value=0.0, value=1.2, step=0.1)
    with col3:
        a_xg = st.number_input("Away xG", min_value=0.0, value=0.5, step=0.1)

    st.markdown("#### 2. Your Position (Betting Slip)")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        stake = st.number_input("Back Stake (£)", value=50.0)
    with col5:
        back_odds = st.number_input("Bookie Odds", value=2.50)
    with col6:
        # Crucial: The odds available NOW to BACK the leader (to close lay)
        current_odds = st.number_input("Current Live Odds", value=1.10, help="The current price to Back the leading team on the Exchange")

# --- ANALYSIS SECTION ---
if st.button("Analyze & Calculate Strategy", type="primary"):
    
    # A. AI Prediction
    prob = calculate_probability(minute, h_xg, a_xg, model, scaler)
    
    # B. Financials
    cash_out_profit, max_upside = calculate_financials(stake, back_odds, current_odds, 0.02)
    
    st.divider()
    
    # --- VISUALIZING THE DECISION ---
    
    # Thresholds
    THRESHOLD_BET = 0.05
    
    st.subheader("🤖 AI Verdict")
    
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Comeback Probability", f"{prob:.1%}", delta_color="inverse")
    
    if prob >= THRESHOLD_BET:
        decision_color = "green"
        decision_text = "LET IT RIDE (Don't Cash Out)"
        reason = "The AI detects statistical fragility in this lead. The risk/reward favors holding."
    else:
        decision_color = "red"
        decision_text = "CASH OUT NOW (Take Profit)"
        reason = "The lead is statistically secure. Secure your profit immediately."
        
    st.markdown(f":{decision_color}[**DECISION: {decision_text}**]")
    st.info(reason)

    # --- FINANCIAL BREAKDOWN ---
    st.subheader("💰 Financial Options")
    
    fin_col1, fin_col2 = st.columns(2)
    
    with fin_col1:
        st.markdown("##### Option A: Safe Exit")
        st.markdown(f"**Profit Now:** £{cash_out_profit:.2f}")
        st.caption("You close the trade on the exchange immediately.")
        if prob < THRESHOLD_BET:
             st.success("Recommended Option")

    with fin_col2:
        st.markdown("##### Option B: Full Turnaround")
        st.markdown(f"**Potential Profit:** £{max_upside:.2f}")
        st.caption("You wait for a Draw or Loss. If Leader wins, you get £0 extra.")
        if prob >= THRESHOLD_BET:
             st.success("Recommended Option")

    # Visualizing Expected Value
    ev_ride = (prob * max_upside) + ((1-prob) * 0) # Simplified EV of waiting
    
    st.write("---")
    st.markdown("#### Mathematical Expected Value (EV)")
    st.bar_chart(pd.DataFrame({
        'Strategy Value': [cash_out_profit, ev_ride]
    }, index=['Cash Out Now', 'Expected Value of Waiting']))