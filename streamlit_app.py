import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import joblib
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import schedule
import threading
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json

# Page configuration
st.set_page_config(page_title="Smart Job Matcher", layout="wide", page_icon="🎯")

# Your existing scraping function
def scrape_karkidi_jobs(keywords=["data scientist"], pages=2):
    """Your existing scraping function"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for keyword in keywords:
        for page in range(1, pages + 1):
            url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
            print(f"Scraping page {page} for '{keyword}'...")
            try:
                response = requests.get(url, headers=headers, timeout=10)
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
                            "Skills": skills,
                            "Scraped_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    except Exception as e:
                        print(f"Error parsing job block: {e}")
                        continue

                time.sleep(1)
            except Exception as e:
                st.error(f"Error scraping page {page}: {e}")
                continue

    return pd.DataFrame(jobs_list)

# Your clustering and saving function (modified)
def cluster_and_save(X, df, tfidf, k):
    """Your existing clustering function with minor modifications"""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df["Cluster"] = kmeans.fit_predict(X)

    # Save the model and vectorizer
    joblib.dump(kmeans, "kmeans_model.joblib")
    joblib.dump(tfidf, "tfidf_vectorizer.joblib")
    df.to_csv("clustered_jobs.csv", index=False)

    st.success("✅ Model and vectorizer saved. Clustered data exported to 'clustered_jobs.csv'")
    return df

def load_saved_models():
    """Load your saved models"""
    try:
        if os.path.exists("kmeans_model.joblib") and os.path.exists("tfidf_vectorizer.joblib"):
            kmeans = joblib.load("kmeans_model.joblib")
            tfidf = joblib.load("tfidf_vectorizer.joblib")
            
            if os.path.exists("clustered_jobs.csv"):
                df = pd.read_csv("clustered_jobs.csv")
                return kmeans, tfidf, df
            else:
                return kmeans, tfidf, None
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def prepare_text_for_clustering(df):
    """Prepare combined text for clustering"""
    df['combined_text'] = (
        df['Title'].fillna('') + ' ' + 
        df['Skills'].fillna('') + ' ' + 
        df['Summary'].fillna('')
    )
    return df

class JobMatcher:
    def __init__(self, kmeans_model, tfidf_vectorizer, jobs_df):
        self.kmeans = kmeans_model
        self.tfidf = tfidf_vectorizer
        self.jobs_df = jobs_df
        
    def find_similar_jobs(self, user_interests, top_n=5):
        """Find jobs similar to user interests"""
        try:
            # Transform user interests using saved TF-IDF
            user_vector = self.tfidf.transform([user_interests])
            
            # Get job vectors
            job_vectors = self.tfidf.transform(self.jobs_df['combined_text'])
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()
            
            # Get top similar jobs
            top_indices = similarity_scores.argsort()[-top_n:][::-1]
            similar_jobs = self.jobs_df.iloc[top_indices].copy()
            similar_jobs['Similarity_Score'] = similarity_scores[top_indices]
            
            return similar_jobs
        except Exception as e:
            st.error(f"Error finding similar jobs: {e}")
            return pd.DataFrame()
    
    def predict_cluster(self, user_interests):
        """Predict cluster for user interests"""
        try:
            user_vector = self.tfidf.transform([user_interests])
            cluster = self.kmeans.predict(user_vector)[0]
            return cluster
        except Exception as e:
            st.error(f"Error predicting cluster: {e}")
            return None

class UserPreferences:
    def __init__(self):
        self.preferences_file = "user_preferences.json"
        
    def load_preferences(self):
        """Load user preferences"""
        if os.path.exists(self.preferences_file):
            with open(self.preferences_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_preferences(self, preferences):
        """Save user preferences"""
        with open(self.preferences_file, 'w') as f:
            json.dump(preferences, f, indent=2)

class NotificationSystem:
    def __init__(self, job_matcher, user_prefs):
        self.job_matcher = job_matcher
        self.user_prefs = user_prefs
        
    def check_new_matches(self, new_jobs_df):
        """Check for new job matches based on user preferences"""
        notifications = []
        preferences = self.user_prefs.load_preferences()
        
        for user_email, prefs in preferences.items():
            if prefs.get('notifications_enabled', False):
                # Prepare new jobs for matching
                new_jobs_prepared = prepare_text_for_clustering(new_jobs_df)
                
                # Create temporary matcher for new jobs
                temp_matcher = JobMatcher(self.job_matcher.kmeans, self.job_matcher.tfidf, new_jobs_prepared)
                
                # Find similar jobs
                similar_jobs = temp_matcher.find_similar_jobs(prefs['interests'], top_n=5)
                
                if not similar_jobs.empty:
                    # Filter high-match jobs
                    high_match_jobs = similar_jobs[similar_jobs['Similarity_Score'] > 0.3]
                    if not high_match_jobs.empty:
                        notifications.append({
                            'user_email': user_email,
                            'jobs': high_match_jobs,
                            'preferences': prefs
                        })
        
        return notifications
    
    def send_email_notification(self, user_email, jobs_df, smtp_config):
        """Send email notification"""
        try:
            msg = MimeMultipart()
            msg['From'] = smtp_config['email']
            msg['To'] = user_email
            msg['Subject'] = f"🎯 {len(jobs_df)} New Job Matches Found!"
            
            # Create email body
            body = self.create_email_body(jobs_df)
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['port'])
            server.starttls()
            server.login(smtp_config['email'], smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            st.error(f"Failed to send email: {e}")
            return False
    
    def create_email_body(self, jobs_df):
        """Create HTML email body"""
        html = """
        <html><body>
        <h2>🎯 New Job Matches Found!</h2>
        <p>Here are some exciting opportunities that match your interests:</p>
        """
        
        for _, job in jobs_df.iterrows():
            html += f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0;">
                <h3>{job['Title']}</h3>
                <p><strong>Company:</strong> {job['Company']}</p>
                <p><strong>Location:</strong> {job['Location']}</p>
                <p><strong>Experience:</strong> {job['Experience']}</p>
                <p><strong>Match Score:</strong> {job['Similarity_Score']:.1%}</p>
                <p><strong>Skills:</strong> {job['Skills'][:200]}...</p>
            </div>
            """
        
        html += "</body></html>"
        return html

class AutomationScheduler:
    def __init__(self, job_matcher, notification_system):
        self.job_matcher = job_matcher
        self.notification_system = notification_system
        self.is_running = False
        
    def daily_scrape_and_notify(self):
        """Daily automated scraping and notification"""
        try:
            st.info("🤖 Running automated daily scrape...")
            
            # Scrape new jobs
            keywords = ["data scientist", "machine learning", "data analyst", "AI engineer"]
            new_jobs = scrape_karkidi_jobs(keywords=keywords, pages=2)
            
            if not new_jobs.empty:
                # Check for matches and send notifications
                notifications = self.notification_system.check_new_matches(new_jobs)
                
                # Save new jobs to history
                self.save_to_history(new_jobs)
                
                st.success(f"✅ Scraped {len(new_jobs)} jobs, found {len(notifications)} user matches")
                return new_jobs, notifications
            else:
                st.warning("⚠️ No new jobs found")
                return pd.DataFrame(), []
                
        except Exception as e:
            st.error(f"❌ Error in automated scraping: {e}")
            return pd.DataFrame(), []
    
    def save_to_history(self, jobs_df):
        """Save jobs to historical data"""
        history_file = 'jobs_history.csv'
        if os.path.exists(history_file):
            existing = pd.read_csv(history_file)
            combined = pd.concat([existing, jobs_df], ignore_index=True)
        else:
            combined = jobs_df
        
        # Keep only last 30 days
        combined['Scraped_Date'] = pd.to_datetime(combined['Scraped_Date'])
        cutoff = datetime.now() - timedelta(days=30)
        combined = combined[combined['Scraped_Date'] >= cutoff]
        combined.to_csv(history_file, index=False)

def main():
    st.title("🎯 Smart Job Matcher with Saved Models")
    st.markdown("*Using your pre-trained K-means clustering and TF-IDF models*")
    
    # Load saved models
    kmeans_model, tfidf_vectorizer, jobs_df = load_saved_models()
    
    # Initialize components
    user_prefs = UserPreferences()
    
    if kmeans_model is not None and tfidf_vectorizer is not None:
        if jobs_df is not None:
            # Prepare jobs data
            jobs_df = prepare_text_for_clustering(jobs_df)
            job_matcher = JobMatcher(kmeans_model, tfidf_vectorizer, jobs_df)
            notification_system = NotificationSystem(job_matcher, user_prefs)
            scheduler = AutomationScheduler(job_matcher, notification_system)
            
            # Sidebar navigation
            st.sidebar.title("Navigation")
            page = st.sidebar.selectbox("Choose a page", [
                "🏠 Home", 
                "🔍 Job Matching", 
                "👤 User Preferences", 
                "🔔 Notifications",
                "🤖 Automation",
                "📊 Model Info"
            ])
            
            if page == "🏠 Home":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Jobs in Database", len(jobs_df))
                with col2:
                    st.metric("Clusters", len(jobs_df['Cluster'].unique()))
                with col3:
                    st.metric("Companies", jobs_df['Company'].nunique())
                
                st.subheader("🎯 Quick Job Match")
                user_input = st.text_input("Enter your skills:", placeholder="python, machine learning, data science...")
                
                if user_input and st.button("Find Matches"):
                    similar_jobs = job_matcher.find_similar_jobs(user_input, top_n=3)
                    predicted_cluster = job_matcher.predict_cluster(user_input)
                    
                    st.success(f"🎯 You belong to Cluster {predicted_cluster}")
                    
                    for _, job in similar_jobs.iterrows():
                        with st.expander(f"🏢 {job['Title']} - {job['Company']} ({job['Similarity_Score']:.1%} match)"):
                            st.write(f"**Location:** {job['Location']}")
                            st.write(f"**Experience:** {job['Experience']}")
                            st.write(f"**Skills:** {job['Skills'][:200]}...")
            
            elif page == "🔍 Job Matching":
                st.header("Advanced Job Matching")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🎯 Find Your Perfect Match")
                    user_interests = st.text_area("Your Skills & Interests:", 
                                                height=100,
                                                placeholder="machine learning, python, deep learning, NLP, computer vision...")
                    
                    num_results = st.slider("Number of results", 1, 20, 10)
                    
                    if st.button("🔍 Find Matching Jobs", type="primary") and user_interests:
                        similar_jobs = job_matcher.find_similar_jobs(user_interests, top_n=num_results)
                        predicted_cluster = job_matcher.predict_cluster(user_interests)
                        
                        st.success(f"🎯 **Predicted Cluster: {predicted_cluster}**")
                        
                        if not similar_jobs.empty:
                            for idx, job in similar_jobs.iterrows():
                                score = job['Similarity_Score']
                                color = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
                                
                                with st.expander(f"{color} {job['Title']} - {job['Company']} ({score:.1%})"):
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.write(f"**📍 Location:** {job['Location']}")
                                        st.write(f"**💼 Experience:** {job['Experience']}")
                                    with col_b:
                                        st.write(f"**🎯 Cluster:** {job['Cluster']}")
                                        st.write(f"**📊 Match Score:** {score:.1%}")
                                    
                                    st.write(f"**🔧 Skills:** {job['Skills']}")
                                    st.write(f"**📝 Summary:** {job['Summary'][:300]}...")
                
                with col2:
                    st.subheader("📊 Cluster Analysis")
                    cluster_counts = jobs_df['Cluster'].value_counts().sort_index()
                    st.bar_chart(cluster_counts)
                    
                    st.subheader("🏢 Top Companies")
                    company_counts = jobs_df['Company'].value_counts().head(10)
                    st.bar_chart(company_counts)
            
            elif page == "👤 User Preferences":
                st.header("User Preferences & Interest Tracking")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📝 Set Your Preferences")
                    
                    user_email = st.text_input("Email Address")
                    user_interests = st.text_area("Your Interests & Skills", height=100)
                    preferred_locations = st.text_input("Preferred Locations")
                    experience_level = st.selectbox("Experience Level", 
                                                  ["0-2 years", "2-5 years", "5-8 years", "8+ years"])
                    notifications_enabled = st.checkbox("Enable Email Notifications", value=True)
                    
                    if st.button("💾 Save Preferences", type="primary"):
                        if user_email and user_interests:
                            preferences = user_prefs.load_preferences()
                            preferences[user_email] = {
                                'interests': user_interests,
                                'locations': preferred_locations,
                                'experience': experience_level,
                                'notifications_enabled': notifications_enabled,
                                'created_date': datetime.now().isoformat()
                            }
                            user_prefs.save_preferences(preferences)
                            st.success("✅ Preferences saved!")
                        else:
                            st.error("❌ Please fill email and interests")
                
                with col2:
                    st.subheader("👥 Registered Users")
                    preferences = user_prefs.load_preferences()
                    
                    if preferences:
                        for email, prefs in preferences.items():
                            with st.expander(f"📧 {email}"):
                                st.write(f"**Interests:** {prefs['interests'][:100]}...")
                                st.write(f"**Notifications:** {'✅' if prefs['notifications_enabled'] else '❌'}")
                    else:
                        st.info("No users registered yet")
            
            elif page == "🔔 Notifications":
                st.header("Email Notification System")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📧 Email Configuration")
                    smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
                    smtp_port = st.number_input("Port", value=587)
                    sender_email = st.text_input("Sender Email")
                    sender_password = st.text_input("Password", type="password")
                    
                    if st.button("🧪 Test Configuration"):
                        if all([smtp_server, smtp_port, sender_email, sender_password]):
                            try:
                                server = smtplib.SMTP(smtp_server, smtp_port)
                                server.starttls()
                                server.login(sender_email, sender_password)
                                server.quit()
                                st.success("✅ Email configuration successful!")
                                st.session_state['smtp_config'] = {
                                    'smtp_server': smtp_server,
                                    'port': smtp_port,
                                    'email': sender_email,
                                    'password': sender_password
                                }
                            except Exception as e:
                                st.error(f"❌ Configuration failed: {e}")
                
                with col2:
                    st.subheader("🔔 Send Test Notifications")
                    
                    if st.button("Send Notifications") and 'smtp_config' in st.session_state:
                        # Simulate new jobs for testing
                        test_jobs = jobs_df.sample(3) if len(jobs_df) > 3 else jobs_df
                        notifications = notification_system.check_new_matches(test_jobs)
                        
                        for notification in notifications:
                            success = notification_system.send_email_notification(
                                notification['user_email'],
                                notification['jobs'],
                                st.session_state['smtp_config']
                            )
                            if success:
                                st.success(f"✅ Sent to {notification['user_email']}")
            
            elif page == "🤖 Automation":
                st.header("Automated Daily Scraping")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("⏰ Automation Status")
                    
                    if scheduler.is_running:
                        st.success("🟢 Automation RUNNING")
                        if st.button("⏹️ Stop Automation"):
                            scheduler.is_running = False
                            st.rerun()
                    else:
                        st.info("🔴 Automation STOPPED")
                        if st.button("▶️ Start Automation"):
                            scheduler.is_running = True
                            # Start scheduler in background
                            schedule.every().day.at("09:00").do(scheduler.daily_scrape_and_notify)
                            st.rerun()
                    
                    if st.button("🚀 Run Manual Scrape"):
                        new_jobs, notifications = scheduler.daily_scrape_and_notify()
                        if not new_jobs.empty:
                            st.success(f"Found {len(new_jobs)} new jobs!")
                
                with col2:
                    st.subheader("📊 Scraping History")
                    if os.path.exists('jobs_history.csv'):
                        history = pd.read_csv('jobs_history.csv')
                        st.write(f"**Total historical jobs:** {len(history)}")
                        
                        # Show recent activity
                        history['Scraped_Date'] = pd.to_datetime(history['Scraped_Date'])
                        daily_counts = history.groupby(history['Scraped_Date'].dt.date).size()
                        st.line_chart(daily_counts.tail(7))
            
            elif page == "📊 Model Info":
                st.header("Saved Model Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🤖 K-means Model")
                    st.write(f"**Number of clusters:** {kmeans_model.n_clusters}")
                    st.write(f"**Random state:** {kmeans_model.random_state}")
                    st.write(f"**Algorithm:** {kmeans_model.algorithm}")
                    
                    st.subheader("📊 Cluster Distribution")
                    cluster_counts = jobs_df['Cluster'].value_counts().sort_index()
                    st.bar_chart(cluster_counts)
                
                with col2:
                    st.subheader("📝 TF-IDF Vectorizer")
                    st.write(f"**Max features:** {tfidf_vectorizer.max_features}")
                    st.write(f"**Vocabulary size:** {len(tfidf_vectorizer.vocabulary_)}")
                    st.write(f"**Stop words:** {tfidf_vectorizer.stop_words}")
                    
                    st.subheader("💾 File Information")
                    files_info = {
                        "kmeans_model.joblib": os.path.getsize("kmeans_model.joblib") if os.path.exists("kmeans_model.joblib") else 0,
                        "tfidf_vectorizer.joblib": os.path.getsize("tfidf_vectorizer.joblib") if os.path.exists("tfidf_vectorizer.joblib") else 0,
                        "clustered_jobs.csv": os.path.getsize("clustered_jobs.csv") if os.path.exists("clustered_jobs.csv") else 0
                    }
                    
                    for file, size in files_info.items():
                        st.write(f"**{file}:** {size/1024:.1f} KB")
        
        else:
            st.warning("⚠️ Models found but no job data. Please run clustering first.")
            
            if st.button("🔄 Scrape New Data and Cluster"):
                with st.spinner("Scraping jobs..."):
                    df_jobs = scrape_karkidi_jobs(["data scientist", "machine learning"], pages=2)
                
                if not df_jobs.empty:
                    with st.spinner("Clustering jobs..."):
                        df_prepared = prepare_text_for_clustering(df_jobs)
                        X = tfidf_vectorizer.fit_transform(df_prepared['combined_text'])
                        df_clustered = cluster_and_save(X, df_prepared, tfidf_vectorizer, 5)
                    st.rerun()
    
    else:
        st.error("❌ No saved models found!")
        st.markdown("""
        ### 🚀 Getting Started:
        1. **Train Models First**: Run your Jupyter notebook to create the models
        2. **Required Files**:
           - `kmeans_model.joblib`
           - `tfidf_vectorizer.joblib` 
           - `clustered_jobs.csv`
        3. **Place Files**: Put these files in the same directory as this Streamlit app
        """)
        
        if st.button("🔄 Check for Models Again"):
            st.rerun()

if __name__ == "__main__":
    main()
