# fusion_model.py
import joblib
import numpy as np
from live_detector import preprocess_text
import datetime

print("Loading models...")
audio_model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")
txn_model = joblib.load("transaction_model.pkl")
print("✅ All models loaded successfully.\n")

def get_audio_prob(text):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    prob = audio_model.predict_proba(text_vector)[0][1]
    return prob

def get_txn_prob(transaction_features):
    try:
        txn_features = np.array([transaction_features], dtype=float)
        prob = txn_model.predict_proba(txn_features)[0][1]
    except (AttributeError, IndexError, TypeError) as e:
        print(f"⚠️ predict_proba failed: {e}. Using predict() fallback.")
        txn_features = np.array([transaction_features], dtype=float)
        pred = txn_model.predict(txn_features)
        if isinstance(pred, (list, np.ndarray)):
            prob = float(pred[0]) if len(pred) > 0 else 0.5
        else:
            prob = float(pred)
        prob = max(0.0, min(1.0, prob))
    return prob

def unified_prediction(audio_text=None, transaction_features=None, w_audio=0.6, w_txn=0.4):
    if audio_text and transaction_features:
        audio_prob = get_audio_prob(audio_text)
        txn_prob = get_txn_prob(transaction_features)
        final_prob = w_audio * audio_prob + w_txn * txn_prob
        source = "Audio + Transaction"
    elif audio_text:
        audio_prob = get_audio_prob(audio_text)
        final_prob = audio_prob
        txn_prob = None
        source = "Audio Only"
    elif transaction_features is not None:
        txn_prob = get_txn_prob(transaction_features)
        final_prob = txn_prob
        audio_prob = None
        source = "Transaction Only"
    else:
        raise ValueError("❌ No input provided! Please give either audio_text or transaction_features.")

    hour = datetime.datetime.now().hour
    print(f"🕒 Current system hour detected: {hour}")

    # ✅ FIXED: Correct time adjustments
    if hour >= 20 or hour <= 5:
        print("🌙 Late-night transaction detected — increasing fraud risk.")
        final_prob = min(1.0, final_prob + 0.2)  # ✅ ADD for late night
    elif 9 <= hour <= 18:
        print("🌞 Business hours — reducing fraud probability.")
        final_prob = max(0.0, final_prob - 0.1)  # ✅ SUBTRACT for business hours
    else:
        print("🕒 Neutral hour — no adjustment applied.")

    print(f"\n📡 Input Source: {source}")
    if audio_prob is not None:
        print(f"🎧 Audio Model Fraud Probability: {audio_prob:.2f}")
    if txn_prob is not None:
        print(f"💳 Transaction Model Fraud Probability: {txn_prob:.2f}")

    print(f"🔗 Final Fraud Probability: {final_prob:.2f}")

    if final_prob > 0.5:
        print("\n🚨 ALERT: Fraud Detected!")
        label = 1
    else:
        print("\n✅ Transaction Normal.")
        label = 0

    details = {
        "final_fraud_probability": final_prob,
        "is_fraud": (label == 1),
        "details": {
            "source": source,
            "base_audio_prob": audio_prob,
            "base_txn_prob": txn_prob,
            "time_adjustment_reason": "Time-based adjustment applied",
            "time_adjustment_value": 0.0,
            "weights": {
                "audio": w_audio,
                "transaction": w_txn
            }
        }
    }
    return details
