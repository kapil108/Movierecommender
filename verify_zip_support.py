import pandas as pd
import os
import sys

def verify_zip_reading():
    print("Verifying ZIP reading capability...")
    
    zip_file = "tmdb_5000_credits.zip"
    csv_file = "tmdb_5000_credits.csv"
    
    if not os.path.exists(zip_file):
        print(f"ERROR: {zip_file} not found. Cannot verify.")
        return

    # Rename CSV to force reading from ZIP
    renamed = False
    if os.path.exists(csv_file):
        print(f"Temporarily hiding {csv_file}...")
        try:
            os.rename(csv_file, csv_file + ".bak")
            renamed = True
        except Exception as e:
            print(f"Error hiding CSV: {e}")
            return

    try:
        print(f"Attempting to read {zip_file} using pandas...")
        # Simulating the logic in app.py
        if os.path.exists(zip_file):
             df = pd.read_csv(zip_file)
             print(f"SUCCESS: Read dataframe from zip. Shape: {df.shape}")
             print(f"Columns: {list(df.columns)[:5]}")
        else:
             print("ERROR: Zip file logic failed.")
        
    except Exception as e:
        print(f"FAILURE: Could not read zip file. Error: {e}")
    finally:
        if renamed:
            print(f"Restoring {csv_file}...")
            os.rename(csv_file + ".bak", csv_file)

if __name__ == "__main__":
    verify_zip_reading()
