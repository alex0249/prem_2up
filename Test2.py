import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix

# Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')

def analyze_thresholds():
    # 1. Load Data (Skip scraping)
    try:
        df = pd.read_csv("epl_2up_full_history.csv")
        logging.info(f"Loaded {len(df)} matches from history.")
    except FileNotFoundError:
        logging.error("CSV not found. Please run the scraper code first.")
        return

    # 2. Feature Engineering (Same as before)
    df['early_lead_factor'] = (90 - df['minute_2_0']) * df['xg_at_event']
    
    features = ['minute_2_0', 'xg_at_event', 'early_lead_factor']
    
    # Chronological Split
    train = df[df['year'] < 2022]
    test = df[df['year'] >= 2022]
    
    X_train = train[features]
    y_train = train['target_comeback']
    X_test = test[features]
    y_test = test['target_comeback']

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train SVM
    # Note: Increased 'C' to 10.0 to penalize mistakes more heavily
    logging.info("Training SVM...")
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)

    # 5. Get Probabilities (The raw confidence scores)
    y_probs = svm.predict_proba(X_test_scaled)[:, 1]

    # 6. Analyze Thresholds
    # We test thresholds from 0.05 to 0.50
    thresholds = np.arange(0.05, 0.55, 0.01)
    best_f1 = 0
    best_thresh = 0
    best_profit = -float('inf')
    best_profit_thresh = 0

    results = []

    print(f"\n{'Threshold':<10} | {'Bets':<6} | {'Wins':<6} | {'Precision':<10} | {'Recall':<10} | {'Simulated Profit (Units)':<10}")
    print("-" * 80)

    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        
        # Calculate Metrics
        cm = confusion_matrix(y_test, preds)
        # cm structure: [[TN, FP], [FN, TP]]
        tp = cm[1, 1] # Winning Bets
        fp = cm[0, 1] # Losing Bets
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + 22) # 22 is total comebacks in test set
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate Betting Profit
        # Assumption: Average odds for a 2-0 comeback are roughly 20.0
        # Cost per bet = 1 unit. Win = 19 units profit. Loss = -1 unit.
        profit = (tp * 19) - (fp * 1)
        
        results.append({
            'thresh': thresh,
            'f1': f1,
            'profit': profit
        })

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
        
        if profit > best_profit:
            best_profit = profit
            best_profit_thresh = thresh

        # Print distinct steps to show progression
        if int(thresh * 100) % 5 == 0:
            print(f"{thresh:.2f}       | {tp+fp:<6} | {tp:<6} | {precision:.3f}      | {recall:.3f}      | {profit:.1f}")

    print("-" * 80)
    print(f"\nOPTIMAL STATISTICAL THRESHOLD (Best F1): {best_thresh:.2f}")
    print(f"OPTIMAL BETTING THRESHOLD (Best Profit): {best_profit_thresh:.2f}")
    print(f"Max Potential Profit: {best_profit} units")
    
    # 7. Sanity Check: Plot Histogram of Probabilities
    plt.figure(figsize=(10, 5))
    plt.hist(y_probs, bins=50, alpha=0.7, color='blue', label='Predicted Probabilities')
    plt.axvline(best_profit_thresh, color='red', linestyle='dashed', linewidth=2, label='Best Betting Threshold')
    plt.title("Distribution of Model Confidence Scores")
    plt.xlabel("Probability of Comeback")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    analyze_thresholds()