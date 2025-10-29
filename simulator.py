import requests
import time
import random
from supabase import create_client, Client
import json

# (Supabase connection code is the same)
SUPABASE_URL = "https://fnmoqxenjmkwxkftyoea.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZubW9xeGVuam1rd3hrZnR5b2VhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTU5NTUyMSwiZXhwIjoyMDc3MTcxNTIxfQ.mTRWfnnFWpw69fG6ayYR81MbZr5CGFUq-DHrXux4VZ8" 
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

API_URL = "http://127.0.0.1:8000/check_fraud"

print("--- Transaction Simulator is RUNNING ---")
print("Press Ctrl+C to stop.")

current_step = 1 # Start simulation step

while True:
    try:
        # 1. Create a fake transaction
        fake_user = f"user_{random.randint(1, 100)}"
        
        # Determine type and amount based on fraud probability
        if random.random() > 0.8: 
            fake_amount = random.randint(5000, 25000)
            fake_time = f"{random.randint(0, 5):02d}:{random.randint(0, 59):02d}"
            # Fraud happens in CASH_OUT and TRANSFER [cite: 850, 1422]
            fake_type = random.choice(["CASH_OUT", "TRANSFER"]) 
        else:
            fake_amount = random.randint(10, 1000)
            fake_time = f"{random.randint(9, 18):02d}:{random.randint(0, 59):02d}"
            fake_type = random.choice(["PAYMENT", "CASH_IN", "DEBIT"]) # Less likely fraud types

        # 2. Prepare data for the API (NOW INCLUDES STEP and TYPE)
        payload = {
            "amount": float(fake_amount), # Ensure amount is float
            "time_of_day": fake_time,
            "user_id": fake_user,
            "audio_file_path": "simulated_call.wav",
            "step": current_step, # Add the current step
            "type": fake_type     # Add the transaction type
        }

        # 3. Send transaction to the ML API
        print(f"Step {current_step}: Sending {fake_type} from {fake_user} for ${fake_amount}...")
        response = requests.post(API_URL, json=payload)
        
        # (Rest of the code is the same: check response, save if fraud)
        if response.status_code == 200:
            results = response.json()
            is_fraud = results['is_fraud'] 
            score = results['final_fraud_probability']

            if is_fraud: 
                print(f"!!! FRAUD DETECTED ({score*100:.0f}%)! Saving to database...")
                supabase.table('fraud_alerts').insert({
                    'user_id': fake_user,
                    'amount': fake_amount,
                    'fraud_score': score,
                    'details': results 
                }).execute()
            else:
                print(f"Transaction OK ({score*100:.0f}%)")
        
        else:
            print(f"API Error ({response.status_code}): {response.text}")
        
        current_step += 1 # Increment the step for the next loop
        time.sleep(random.randint(2, 5))

    except requests.exceptions.ConnectionError:
        print("--- API Server is OFFLINE. Is main.py running? Retrying in 10s... ---")
        time.sleep(10)
    except KeyboardInterrupt:
        print("\nStopping simulator.")
        break
    except Exception as e: # Catch other potential errors during simulation
         print(f"--- Simulator Error: {e}. Skipping transaction. ---")
         time.sleep(5)