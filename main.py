# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback 
import random
import numpy as np # Needed for scaling

# --- 1. IMPORT YOUR "BRAIN" ---
from fusion_model import unified_prediction
# --- ALSO IMPORT THE SCALER ---
# You need the scaler that was used during training
# It's likely saved in a file, or you need to recreate it.
# Assuming you saved it like this: joblib.dump(scaler, 'scaler.joblib')
try:
    import joblib
    scaler = joblib.load('scaler.joblib') 
    print("✅ Scaler loaded.")
except FileNotFoundError:
    print("⚠️ WARNING: scaler.joblib not found. Using dummy scaling. Predictions might be inaccurate.")
    # Create a dummy scaler if the file isn't found (for testing only)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # Fit it on some dummy data matching the notebook's features
    dummy_data = np.random.rand(10, 3) # amount, diff_org, diff_dest
    scaler.fit(dummy_data)
except Exception as e:
     print(f"🚨 Error loading scaler: {e}. Using dummy scaling.")
     from sklearn.preprocessing import MinMaxScaler
     scaler = MinMaxScaler()
     dummy_data = np.random.rand(10, 3)
     scaler.fit(dummy_data)


app = FastAPI()

# --- 2. DEFINE THE DATA WE EXPECT FROM THE SIMULATOR ---
# We now need 'step' and 'type' as well
class TransactionData(BaseModel):
    amount: float
    time_of_day: str 
    user_id: str
    audio_file_path: str 
    step: int # Add the step
    type: str # Add the transaction type (e.g., "CASH_OUT", "TRANSFER")

# --- 3. CREATE THE API ENDPOINT (FINAL VERSION) ---
@app.post("/check_fraud")
def check_fraud(data: TransactionData):
    
    try:
        # --- 4. PREPARE THE 8 FEATURES ---
        
        # A. Create balance diffs (We don't have old/new, so we estimate)
        # Rule: Amount is usually subtracted from origin, added to dest
        balance_diff_org_est = -data.amount 
        balance_diff_dest_est = data.amount

        # B. Scale the numerical features (amount, diffs) using the loaded scaler
        # The scaler expects a 2D array: [[amount, diff_org, diff_dest]]
        numerical_features_to_scale = np.array([[data.amount, balance_diff_org_est, balance_diff_dest_est]])
        scaled_features = scaler.transform(numerical_features_to_scale)
        
        scaled_amount = scaled_features[0, 0]
        scaled_diff_org = scaled_features[0, 1]
        scaled_diff_dest = scaled_features[0, 2]

        # C. One-Hot Encode the 'type'
        type_cash_out = 1 if data.type == "CASH_OUT" else 0
        type_debit = 1 if data.type == "DEBIT" else 0
        type_payment = 1 if data.type == "PAYMENT" else 0
        type_transfer = 1 if data.type == "TRANSFER" else 0
        # Note: We assume 'CASH_IN' was the dropped category based on get_dummies(drop_first=True) [cite: 1562]

        # D. Assemble the final 8 features in the correct order
        txn_features = [
            data.step,
            scaled_amount,
            scaled_diff_org,
            scaled_diff_dest,
            type_cash_out,
            type_debit,
            type_payment,
            type_transfer
        ]
        
        # E. Create placeholder audio text
        audio_phrases = [
            "please approve my transaction for my new house",
            "hi i need to send money to my brother it's an emergency",
            "this is a test call",
            "i am a fraudster stealing all of this money"
        ]
        audio_text_input = random.choice(audio_phrases)
        
        # --- 5. CALL YOUR "BRAIN" ---
        print(f"DEBUG: Sending features to model: {txn_features}") # Print for debugging
        results_dict = unified_prediction(
            audio_text=audio_text_input,
            transaction_features=txn_features # Send the full 8 features
        )
        
        # --- 6. RETURN THE RESULTS ---
        return results_dict

    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("🚨🚨 A CRASH HAPPENED IN main.py (check_fraud) 🚨🚨")
        print(f"THE ERROR IS: {e}")
        traceback.print_exc() 
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")