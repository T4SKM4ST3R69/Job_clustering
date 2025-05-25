import pandas as pd
import joblib
import schedule
import time
import os
from datetime import datetime
from app import scrape_karkidi_jobs, classify_jobs, send_email_alert

# Load model/vectorizer
kmeans_model = joblib.load("kmeans_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# User config
USER_EMAIL = "your_email@gmail.com"
PREFERRED_CLUSTER = 2
KEYWORDS = ["data scientist", "software engineer"]

def job_alert_service():
    print(f"[{datetime.now()}] Running daily job scrape...")
    df = scrape_karkidi_jobs(KEYWORDS, pages=2)
    
    if not df.empty:
        df = classify_jobs(df, vectorizer, kmeans_model)
        df['Date'] = datetime.now().strftime("%Y-%m-%d")

        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/{datetime.now().strftime('%Y-%m-%d')}_jobs.csv"
        df.to_csv(log_path, index=False)

        matches = df[df['Cluster'] == PREFERRED_CLUSTER]
        if not matches.empty:
            send_email_alert(USER_EMAIL, matches)
            print(f"[INFO] Sent alert for {len(matches)} matching job(s).")
        else:
            print("[INFO] No matching jobs found.")
    else:
        print("[INFO] No jobs scraped.")

# Schedule daily at 09:00
schedule.every().day.at("09:00").do(job_alert_service)

print("[INFO] Scheduler started. Waiting for next run...")
while True:
    schedule.run_pending()
    time.sleep(60)
