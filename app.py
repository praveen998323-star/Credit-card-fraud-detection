"""
========================================================
  CREDIT CARD FRAUD DETECTION - FLASK WEB SERVER
========================================================
  This is the backend API that powers the dashboard.

  ENDPOINTS:
  GET  /              → Serves the main dashboard
  GET  /api/stats     → Returns dataset statistics
  POST /api/predict   → Predicts fraud for 1 transaction
  GET  /api/batch     → Batch analyze recent transactions
  GET  /api/features  → Feature importance data
  GET  /api/simulate  → Simulate live transaction stream
========================================================
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import joblib, os, random, time
from datetime import datetime, timedelta

app = Flask(__name__)

# ─────────────────────────────────────────────
# LOAD MODEL & ARTIFACTS ON STARTUP
# ─────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), 'model')
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')

print("🔄 Loading model artifacts...")

try:
    model        = joblib.load(os.path.join(MODEL_DIR, 'fraud_model.pkl'))
    scaler       = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
    feat_imp_df  = pd.read_csv(os.path.join(MODEL_DIR, 'feature_importance.csv'))
    cm_data      = np.load(os.path.join(MODEL_DIR, 'confusion_matrix.npy'))
    model_results= pd.read_csv(os.path.join(MODEL_DIR, 'model_results.csv'))
    dataset      = pd.read_csv(os.path.join(DATASET_DIR, 'credit_card_data.csv'))
    print("✅ All artifacts loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Please run training scripts first!")
    MODEL_LOADED = False

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def generate_transaction():
    """Generate a single synthetic transaction for live demo."""
    is_fraud = random.random() < 0.03   # 3% chance of fraud in demo
    
    if is_fraud:
        v_features = {f'V{i}': float(np.random.randn() + [-3,2.5,-4,3,-2.5,-1.5,-5,2,-2,-4.5,3.5,-6,0.5,-7,0.3,-3.5,-5.5,-2.8,1,0.8,1.5,0.6,-0.5,0.2,0.4,-0.3,1.8,0.9][i-1])
                      for i in range(1, 29)}
        amount = round(random.uniform(1, 500), 2)
    else:
        v_features = {f'V{i}': float(np.random.randn() * 0.8) for i in range(1, 29)}
        amount = round(abs(np.random.lognormal(3.0, 1.5)), 2)
    
    amount  = min(amount, 5000)
    t_time  = random.uniform(0, 172800)
    
    return {**v_features, 'Amount': amount, 'Time': t_time, 'is_fraud_actual': is_fraud}


def predict_transaction(trans_dict):
    """Run fraud prediction on a transaction."""
    if not MODEL_LOADED:
        return {'error': 'Model not loaded'}
    
    # Scale Amount and Time
    amount_time = np.array([[trans_dict['Amount'], trans_dict['Time']]])
    scaled = scaler.transform(amount_time)
    
    # Build feature vector
    features = {}
    for col in feature_cols:
        if col == 'scaled_amount':
            features[col] = scaled[0][0]
        elif col == 'scaled_time':
            features[col] = scaled[0][1]
        else:
            features[col] = trans_dict.get(col, 0.0)
    
    X = np.array([[features[c] for c in feature_cols]])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    return {
        'prediction'  : int(pred),
        'fraud_prob'  : round(float(proba[1]) * 100, 2),
        'legit_prob'  : round(float(proba[0]) * 100, 2),
        'risk_level'  : 'HIGH' if proba[1] > 0.7 else ('MEDIUM' if proba[1] > 0.3 else 'LOW'),
        'is_fraud'    : bool(pred == 1)
    }

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', model_loaded=MODEL_LOADED)


@app.route('/api/stats')
def get_stats():
    """Return overall dataset and model statistics."""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Run training scripts first.'}), 503
    
    total   = len(dataset)
    fraud   = int(dataset['Class'].sum())
    legit   = total - fraud
    
    # Confusion matrix data
    tn, fp, fn, tp = cm_data.ravel()
    
    # Model comparison
    model_comp = model_results.to_dict('records')
    
    return jsonify({
        'dataset': {
            'total_transactions': total,
            'fraud_count'       : fraud,
            'legit_count'       : legit,
            'fraud_percentage'  : round(fraud / total * 100, 3),
            'avg_amount'        : round(float(dataset['Amount'].mean()), 2),
            'max_amount'        : round(float(dataset['Amount'].max()), 2),
            'avg_fraud_amount'  : round(float(dataset[dataset['Class']==1]['Amount'].mean()), 2),
            'avg_legit_amount'  : round(float(dataset[dataset['Class']==0]['Amount'].mean()), 2),
        },
        'model': {
            'confusion_matrix': {
                'true_negative' : int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive' : int(tp),
                'accuracy'      : round((tn + tp) / (tn + fp + fn + tp) * 100, 2),
                'precision'     : round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0,
                'recall'        : round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0,
            },
            'model_comparison' : model_comp,
        }
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict fraud for a single transaction."""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    result = predict_transaction(data)
    result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return jsonify(result)


@app.route('/api/simulate')
def simulate_transactions():
    """Generate a batch of live simulated transactions."""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503
    
    count = int(request.args.get('count', 10))
    count = min(count, 50)  # Max 50 at a time
    
    transactions = []
    for i in range(count):
        trans = generate_transaction()
        result = predict_transaction(trans)
        
        transactions.append({
            'id'          : f"TXN{random.randint(100000, 999999)}",
            'amount'      : round(trans['Amount'], 2),
            'fraud_prob'  : result['fraud_prob'],
            'risk_level'  : result['risk_level'],
            'is_fraud'    : result['is_fraud'],
            'actual_fraud': trans['is_fraud_actual'],
            'timestamp'   : (datetime.now() - timedelta(seconds=i*30)).strftime('%H:%M:%S'),
            'merchant'    : random.choice([
                'Amazon', 'Netflix', 'Walmart', 'Apple Store', 'Uber',
                'Shell Gas', 'Target', 'Best Buy', 'Starbucks', 'Unknown Merchant'
            ]),
            'location'    : random.choice([
                'New York', 'Los Angeles', 'Chicago', 'Houston', 'London',
                'Mumbai', 'Tokyo', 'Frankfurt', 'Sydney', 'Unknown Location'
            ])
        })
    
    return jsonify({'transactions': transactions})


@app.route('/api/features')
def get_features():
    """Return feature importance data."""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503
    
    top_features = feat_imp_df.head(15).to_dict('records')
    return jsonify({'features': top_features})


@app.route('/api/amount_distribution')
def amount_distribution():
    """Return amount distribution for fraud vs legit."""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503
    
    fraud_amounts = dataset[dataset['Class']==1]['Amount'].tolist()[:200]
    legit_sample  = dataset[dataset['Class']==0]['Amount'].sample(200, random_state=42).tolist()
    
    # Bucket amounts into ranges
    buckets = [0, 10, 25, 50, 100, 200, 500, 1000, 5000, 99999]
    labels  = ['<$10', '$10-25', '$25-50', '$50-100', '$100-200', '$200-500', '$500-1K', '$1K-5K', '>$5K']
    
    fraud_hist = np.histogram(fraud_amounts, bins=buckets)[0].tolist()
    legit_hist = np.histogram(legit_sample,  bins=buckets)[0].tolist()
    
    return jsonify({
        'labels'       : labels,
        'fraud_counts' : fraud_hist,
        'legit_counts' : legit_hist
    })


if __name__ == '__main__':
    print()
    print("=" * 55)
    print("  🚀 CREDIT CARD FRAUD DETECTION WEB APP")
    print("=" * 55)
    if MODEL_LOADED:
        print("  ✅ Model loaded and ready!")
    else:
        print("  ⚠️  WARNING: Run training scripts first!")
    print("  🌐 Open: http://127.0.0.1:5000")
    print("  🛑 Stop: Press CTRL+C")
    print("=" * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)
