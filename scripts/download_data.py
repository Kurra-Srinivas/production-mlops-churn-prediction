"""
DATA DOWNLOAD UTILITY
=====================

Downloads the standard IBM Telco Customer Churn dataset into data/raw/Telco-Customer-Churn.csv.
"""

import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Telco-Customer-Churn.csv")


def download_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        print(f"[OK] Dataset already present at: {OUTPUT_FILE}")
        return OUTPUT_FILE

    print(f"[*] Downloading Telco Customer Churn dataset from public source...")
    urllib.request.urlretrieve(DATA_URL, OUTPUT_FILE)
    print(f"[OK] Downloaded dataset to: {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    download_dataset()
