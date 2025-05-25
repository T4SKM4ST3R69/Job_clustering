import streamlit as st
import pandas as pd
import joblib
import yagmail
import requests
import time
import os
from bs4 import BeautifulSoup
from datetime import datetime

# ----------- Scrape Function -----------
def scrape_karkidi_jobs(keywords=["data scientist"], pages=2):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for keyword in keywords:
        for page in range(1, pages + 1):
            url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")

            job_blocks = soup.find_all("div", class_="ads-details")
            for job in job_blocks:
                try:
                    title = job.find("h4").get_text(strip=True)
                    company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                    location = job.find("p").get_text(strip=True)
                    experience = job.find("p", class_="emp-exp").get_text(strip=True)
                    key_skills_tag = job.find("span", string="Key Skills")
                    skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                    summary_tag = job.find("span", string="Summary")
                    summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                    jobs_list.append({
                        "Keyword": keyword,
                        "Title": title,
                        "Company": company,
                        "Location": location,
                        "Experience": experience,
                        "Summary": summary,
                        "Skills": skills
                    })
                except Exception as e:
                    print(f"Error parsing job block: {e}")
                    continue
            time.sleep(1)
    return pd.DataFrame(jobs_list)

# ----------- Classifier Function -----------
def classify_jobs(df, vectorizer, model):
    skill_vectors = vectorizer.transform(df['Skills'])
    clusters = model.predict(skill_vectors)
    df['Cluster'] = clusters
    return df

# ----------- Email Alert Function -----------
def send_email_alert(user_email, matching_jobs):
    try:
        yag = yagmail.SMTP("your_email@gmail.com", "your_app_password")  # Replace these
        content = f"New job(s) matching your preference:\n\n{matching_jobs.to_string(index=False)}"
        yag.send(to=user_email, subject="New Job Alert!", contents=content)
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# ----------- Load Models -----------
kmeans_model = joblib.load("kmeans_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# ----------- UI -----------
st.set_page_config(page_title="Job Clustering & Alerts", layout="wide")
st.title("🧠 Job Clustering & Alerts")

with st.sidebar:
    st.header("Preferences")
    user_email = st.text_input("Your Email")
    preferred_cluster = st.number_input("Preferred Cluster", min_value=0, max_value=10, step=1)
    keywords = st.text_input("Keywords (comma-separated)", value="data scientist,software engineer")
    pages = st.slider("Pages to Scrape", 1, 5, 2)
    send_alert = st.checkbox("Send Email if Matches Found")

if st.button("Scrape & Classify Jobs"):
    keyword_list = [k.strip() for k in keywords.split(",")]
    scraped = scrape_karkidi_jobs(keyword_list, pages=pages)
    
    if not scraped.empty:
        classified = classify_jobs(scraped, vectorizer, kmeans_model)
        st.success(f"{len(classified)} jobs classified.")
        st.dataframe(classified)

        if send_alert and user_email:
            matches = classified[classified['Cluster'] == preferred_cluster]
            if not matches.empty:
                sent = send_email_alert(user_email, matches)
                if sent:
                    st.success(f"Alert sent to {user_email}")
                else:
                    st.error("Failed to send email.")
            else:
                st.info("No matching jobs found.")
    else:
        st.warning("No jobs scraped.")
