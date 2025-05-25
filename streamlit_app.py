import streamlit as st
import pandas as pd
import joblib
import yagmail
import schedule
import time
import threading
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import os
import json
from pathlib import Path
import hashlib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Job Alert & Clustering System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .cluster-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# CRITICAL FIX: Move tokenizer to module level
def custom_tokenizer(text):
    """Custom tokenizer that splits on commas and cleans tokens"""
    if pd.isna(text) or text == "" or text is None:
        return []
    return [token.strip() for token in str(text).split(',') if token.strip()]

def preprocess_skills(skills_text):
    """Preprocess skills text for vectorization"""
    if pd.isna(skills_text) or skills_text == "" or skills_text is None:
        return ""
    # Use regex to clean text properly
    cleaned = re.sub(r'[^a-zA-Z, ]', '', str(skills_text))
    return cleaned.lower()

class JobClusteringSystem:
    def __init__(self):
        self.model_path = "kmeans_model.joblib"
        self.vectorizer_path = "tfidf_vectorizer.joblib"
        self.data_path = "clustered_jobs.csv"
        self.config_path = "config.json"
        self.job_history_path = "job_history.json"
        
        # Load models and data
        self.load_models()
        self.load_config()
        self.load_job_history()
    
    def load_models(self):
        """Load saved models and data with proper error handling"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                self.kmeans_model = joblib.load(self.model_path)
                self.tfidf_vectorizer = joblib.load(self.vectorizer_path)
                st.success("✅ Models loaded successfully!")
            else:
                st.warning("❌ Model files not found. You can train new models from scraped data.")
                self.kmeans_model = None
                self.tfidf_vectorizer = None
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Creating new vectorizer as fallback...")
            self.kmeans_model = None
            self.tfidf_vectorizer = None
    
    def load_config(self):
        """Load email configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
    
    def save_config(self):
        """Save email configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
    
    def load_job_history(self):
        """Load job history for tracking new jobs"""
        if os.path.exists(self.job_history_path):
            with open(self.job_history_path, 'r') as f:
                self.job_history = json.load(f)
        else:
            self.job_history = {"job_hashes": [], "last_check": None}
    
    def save_job_history(self):
        """Save job history"""
        with open(self.job_history_path, 'w') as f:
            json.dump(self.job_history, f)
    
    def generate_job_hash(self, job_data):
        """Generate unique hash for a job listing"""
        job_string = f"{job_data['Title']}{job_data['Company']}{job_data['Location']}"
        return hashlib.md5(job_string.encode()).hexdigest()
    
    def train_clustering_model(self, df, n_clusters=5):
        """Train clustering model on job data"""
        try:
            # Prepare skills data for vectorization
            skills_data = []
            for skills in df['Skills'].fillna(''):
                processed = preprocess_skills(skills)
                skills_data.append(processed)
            
            # Create TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )
            
            # Fit and transform skills data
            skills_matrix = self.tfidf_vectorizer.fit_transform(skills_data)
            
            # Train KMeans model
            self.kmeans_model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            
            # Fit the model
            clusters = self.kmeans_model.fit_predict(skills_matrix)
            
            # Add cluster labels to dataframe
            df['Cluster'] = clusters
            
            # Save models
            joblib.dump(self.kmeans_model, self.model_path)
            joblib.dump(self.tfidf_vectorizer, self.vectorizer_path)
            
            # Save clustered data
            df.to_csv(self.data_path, index=False)
            
            return df, True
            
        except Exception as e:
            st.error(f"Error training clustering model: {str(e)}")
            return df, False
    
    def predict_cluster(self, skills_text):
        """Predict cluster for new job based on skills"""
        if self.kmeans_model is None or self.tfidf_vectorizer is None:
            return -1
        
        try:
            processed_skills = preprocess_skills(skills_text)
            if not processed_skills:
                return -1
            
            skills_vector = self.tfidf_vectorizer.transform([processed_skills])
            cluster = self.kmeans_model.predict(skills_vector)[0]
            return cluster
        except Exception as e:
            st.warning(f"Error predicting cluster: {str(e)}")
            return -1
    
    def scrape_new_jobs(self, keywords=["data scientist"], pages=2):
        """Scrape new jobs from Karkidi"""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
        jobs_list = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_pages = len(keywords) * pages
        current_page = 0
        
        for keyword in keywords:
            for page in range(1, pages + 1):
                try:
                    url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
                    status_text.text(f"Scraping page {page} for '{keyword}'...")
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.content, "html.parser")
                    
                    job_blocks = soup.find_all("div", class_="ads-details")
                    
                    for job in job_blocks:
                        try:
                            title = job.find("h4").get_text(strip=True) if job.find("h4") else "N/A"
                            company_element = job.find("a", href=lambda x: x and "Employer-Profile" in x)
                            company = company_element.get_text(strip=True) if company_element else "N/A"
                            location = job.find("p").get_text(strip=True) if job.find("p") else "N/A"
                            
                            exp_element = job.find("p", class_="emp-exp")
                            experience = exp_element.get_text(strip=True) if exp_element else "N/A"
                            
                            key_skills_tag = job.find("span", string="Key Skills")
                            skills = ""
                            if key_skills_tag:
                                skills_p = key_skills_tag.find_next("p")
                                skills = skills_p.get_text(strip=True) if skills_p else ""
                            
                            summary_tag = job.find("span", string="Summary")
                            summary = ""
                            if summary_tag:
                                summary_p = summary_tag.find_next("p")
                                summary = summary_p.get_text(strip=True) if summary_p else ""
                            
                            job_data = {
                                "Keyword": keyword,
                                "Title": title,
                                "Company": company,
                                "Location": location,
                                "Experience": experience,
                                "Summary": summary,
                                "Skills": skills,
                                "Scraped_At": datetime.now().isoformat()
                            }
                            
                            jobs_list.append(job_data)
                            
                        except Exception as e:
                            continue
                    
                    current_page += 1
                    progress_bar.progress(current_page / total_pages)
                    time.sleep(1)  # Be respectful to the server
                    
                except Exception as e:
                    st.warning(f"Error scraping page {page} for '{keyword}': {str(e)}")
                    continue
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(jobs_list)
    
    def identify_new_jobs(self, df_new):
        """Identify truly new jobs by comparing with history"""
        if df_new.empty:
            return df_new
        
        new_jobs = []
        for _, job in df_new.iterrows():
            job_hash = self.generate_job_hash(job)
            if job_hash not in self.job_history["job_hashes"]:
                new_jobs.append(job)
                self.job_history["job_hashes"].append(job_hash)
        
        # Keep only last 1000 job hashes to prevent memory issues
        if len(self.job_history["job_hashes"]) > 1000:
            self.job_history["job_hashes"] = self.job_history["job_hashes"][-1000:]
        
        self.job_history["last_check"] = datetime.now().isoformat()
        self.save_job_history()
        
        return pd.DataFrame(new_jobs)
    
    def send_email_alert(self, new_jobs_df, user_clusters):
        """Send email alert for new jobs in user's preferred clusters"""
        if new_jobs_df.empty or not self.config.get('email_enabled', False):
            return False
        
        # Add cluster predictions to new jobs
        new_jobs_df['Predicted_Cluster'] = new_jobs_df['Skills'].apply(self.predict_cluster)
        
        # Filter jobs that match user's preferred clusters
        matching_jobs = new_jobs_df[new_jobs_df['Predicted_Cluster'].isin(user_clusters)]
        
        if matching_jobs.empty:
            return False
        
        try:
            # Create email content
            subject = f"🔔 {len(matching_jobs)} New Job Alert(s) - {datetime.now().strftime('%Y-%m-%d')}"
            
            html_content = f"""
            <html>
            <body>
            <h2>New Job Opportunities Found!</h2>
            <p>We found {len(matching_jobs)} new job(s) matching your preferred categories:</p>
            """
            
            for _, job in matching_jobs.iterrows():
                html_content += f"""
                <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                    <h3>{job['Title']}</h3>
                    <p><strong>Company:</strong> {job['Company']}</p>
                    <p><strong>Location:</strong> {job['Location']}</p>
                    <p><strong>Experience:</strong> {job['Experience']}</p>
                    <p><strong>Skills:</strong> {job['Skills']}</p>
                    <p><strong>Cluster:</strong> {job['Predicted_Cluster']}</p>
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            # Send email using yagmail
            yag = yagmail.SMTP(
                self.config['sender_email'], 
                self.config['sender_password']
            )
            
            yag.send(
                to=self.config['recipient_email'],
                subject=subject,
                contents=html_content
            )
            
            return True
            
        except Exception as e:
            st.error(f"Failed to send email: {str(e)}")
            return False

def main():
    # Initialize the system
    job_system = JobClusteringSystem()
    
    # Sidebar for navigation
    st.sidebar.title("🔍 Job Alert System")
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Dashboard", "⚙️ Configuration", "🔄 Manual Scraping", "🤖 Train Clustering", "📧 Email Setup", "📈 Analytics"]
    )
    
    if page == "📊 Dashboard":
        st.markdown("<h1 class='main-header'>Job Clustering Dashboard</h1>", unsafe_allow_html=True)
        
        # Load existing data
        if os.path.exists(job_system.data_path):
            df = pd.read_csv(job_system.data_path)
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Jobs", len(df))
            with col2:
                st.metric("Unique Companies", df['Company'].nunique())
            with col3:
                # Check if Cluster column exists
                if 'Cluster' in df.columns:
                    st.metric("Clusters", df['Cluster'].nunique())
                else:
                    st.metric("Clusters", "Not Trained")
            with col4:
                if job_system.job_history.get('last_check'):
                    last_check = datetime.fromisoformat(job_system.job_history['last_check'])
                    st.metric("Last Check", last_check.strftime("%m/%d %H:%M"))
                else:
                    st.metric("Last Check", "Never")
            
            # Show cluster analysis only if Cluster column exists
            if 'Cluster' in df.columns:
                # Cluster distribution
                st.subheader("📊 Cluster Distribution")
                cluster_counts = df['Cluster'].value_counts().sort_index()
                fig_bar = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    title="Jobs per Cluster",
                    labels={'x': 'Cluster', 'y': 'Number of Jobs'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Detailed cluster view
                st.subheader("🔍 Cluster Details")
                selected_cluster = st.selectbox("Select Cluster to View:", sorted(df['Cluster'].unique()))
                
                cluster_jobs = df[df['Cluster'] == selected_cluster]
                st.write(f"**Jobs in Cluster {selected_cluster}:** {len(cluster_jobs)}")
                
                # Show sample skills for this cluster
                all_skills = []
                for skills in cluster_jobs['Skills'].dropna():
                    all_skills.extend(custom_tokenizer(skills))
                
                if all_skills:
                    skill_counts = pd.Series(all_skills).value_counts().head(10)
                    st.write("**Top Skills in this Cluster:**")
                    st.bar_chart(skill_counts)
                
                # Show sample jobs
                st.write("**Sample Jobs:**")
                st.dataframe(
                    cluster_jobs[['Title', 'Company', 'Location', 'Experience']].head(5),
                    use_container_width=True
                )
            else:
                st.info("💡 No cluster data found. Jobs have been scraped but not yet clustered.")
                st.markdown("### 🎯 Next Steps:")
                st.markdown("1. Go to **🤖 Train Clustering** to create clusters from your job data")
                st.markdown("2. Or scrape more jobs first if you need more data")
            
            # Company distribution
            st.subheader("🏢 Top Companies")
            company_counts = df['Company'].value_counts().head(10)
            fig_company = px.pie(
                values=company_counts.values,
                names=company_counts.index,
                title="Top 10 Companies by Job Count"
            )
            st.plotly_chart(fig_company, use_container_width=True)
            
            # Show data preview
            st.subheader("📋 Recent Jobs")
            st.dataframe(df.head(10), use_container_width=True)
            
        else:
            # SOLUTION FOR "No job data found" - Provide clear guidance
            st.warning("📭 No job data found in the database.")
            st.markdown("""
            ### 🚀 Getting Started
            
            To start using the system, follow these steps:
            
            **Step 1: Scrape Job Data**
            1. Go to the **🔄 Manual Scraping** page
            2. Enter keywords (e.g., "data scientist", "machine learning")
            3. Click "🚀 Start Scraping" to collect job data
            4. Save the scraped jobs to the database
            
            **Step 2: Train Clustering Model**
            1. Go to the **🤖 Train Clustering** page
            2. Choose the number of clusters
            3. Train the model on your scraped data
            
            **Step 3: Configure Alerts**
            1. Set up email notifications
            2. Choose preferred clusters for alerts
            """)
            
            # Quick start button
            if st.button("🚀 Start Scraping Jobs Now"):
                st.rerun()
    
    elif page == "🤖 Train Clustering":
        st.header("🤖 Train Clustering Model")
        
        if os.path.exists(job_system.data_path):
            df = pd.read_csv(job_system.data_path)
            
            st.info(f"Found {len(df)} jobs in database. Ready to train clustering model!")
            
            # Clustering parameters
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=10,
                    value=5,
                    help="Choose how many job categories to create"
                )
            
            with col2:
                st.write("**Current Status:**")
                if 'Cluster' in df.columns:
                    st.success(f"✅ Already clustered into {df['Cluster'].nunique()} groups")
                else:
                    st.warning("❌ No clustering performed yet")
            
            # Preview of skills data
            st.subheader("📋 Skills Data Preview")
            skills_preview = df[['Title', 'Company', 'Skills']].head()
            st.dataframe(skills_preview, use_container_width=True)
            
            # Train clustering model
            if st.button("🎯 Train Clustering Model"):
                with st.spinner("Training clustering model..."):
                    clustered_df, success = job_system.train_clustering_model(df, n_clusters)
                
                if success:
                    st.success("✅ Clustering model trained successfully!")
                    
                    # Show cluster distribution
                    st.subheader("📊 Cluster Results")
                    cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
                    
                    fig = px.bar(
                        x=cluster_counts.index,
                        y=cluster_counts.values,
                        title=f"Distribution of {n_clusters} Job Clusters",
                        labels={'x': 'Cluster ID', 'y': 'Number of Jobs'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show sample from each cluster
                    st.subheader("🔍 Sample Jobs from Each Cluster")
                    for cluster_id in sorted(clustered_df['Cluster'].unique()):
                        with st.expander(f"Cluster {cluster_id} ({len(clustered_df[clustered_df['Cluster'] == cluster_id])} jobs)"):
                            cluster_sample = clustered_df[clustered_df['Cluster'] == cluster_id][['Title', 'Company', 'Skills']].head(3)
                            st.dataframe(cluster_sample, use_container_width=True)
                    
                    st.info("🔄 Go back to Dashboard to see the full clustering analysis!")
                
                else:
                    st.error("❌ Failed to train clustering model. Please check your data.")
        
        else:
            st.warning("📭 No job data found. Please scrape jobs first.")
            st.markdown("Go to **🔄 Manual Scraping** to collect job data before training clusters.")
    
    elif page == "⚙️ Configuration":
        st.header("⚙️ System Configuration")
        
        # Auto-scraping settings
        st.subheader("🔄 Auto-Scraping Settings")
        
        auto_scrape = st.checkbox(
            "Enable Daily Auto-Scraping",
            value=job_system.config.get('auto_scrape', False)
        )
        
        if auto_scrape:
            scrape_time = st.time_input(
                "Daily Scrape Time",
                value=datetime.strptime(job_system.config.get('scrape_time', '09:00'), '%H:%M').time()
            )
            
            keywords = st.text_area(
                "Keywords to Search (one per line)",
                value='\n'.join(job_system.config.get('keywords', ['data scientist']))
            )
            
            pages_to_scrape = st.number_input(
                "Pages to Scrape per Keyword",
                min_value=1,
                max_value=10,
                value=job_system.config.get('pages', 2)
            )
        
        # Cluster preferences
        st.subheader("🎯 Alert Preferences")
        
        if os.path.exists(job_system.data_path):
            df = pd.read_csv(job_system.data_path)
            if 'Cluster' in df.columns:
                available_clusters = sorted(df['Cluster'].unique())
                
                preferred_clusters = st.multiselect(
                    "Select Clusters for Alerts",
                    options=available_clusters,
                    default=job_system.config.get('preferred_clusters', [])
                )
            else:
                preferred_clusters = []
                st.warning("No cluster data found. Please train clustering model first.")
        else:
            preferred_clusters = []
            st.warning("Load job data first to see available clusters")
        
        # Save configuration
        if st.button("💾 Save Configuration"):
            job_system.config.update({
                'auto_scrape': auto_scrape,
                'scrape_time': scrape_time.strftime('%H:%M') if auto_scrape else '09:00',
                'keywords': keywords.split('\n') if auto_scrape else ['data scientist'],
                'pages': pages_to_scrape if auto_scrape else 2,
                'preferred_clusters': preferred_clusters
            })
            job_system.save_config()
            st.success("✅ Configuration saved!")
    
    elif page == "🔄 Manual Scraping":
        st.header("🔄 Manual Job Scraping")
        
        col1, col2 = st.columns(2)
        with col1:
            keywords_input = st.text_area(
                "Keywords (one per line)",
                value="data scientist\nmachine learning engineer\ndata analyst",
                height=100
            )
        
        with col2:
            pages_input = st.number_input(
                "Pages per keyword",
                min_value=1,
                max_value=5,
                value=2
            )
        
        if st.button("🚀 Start Scraping"):
            keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
            
            with st.spinner("Scraping jobs..."):
                new_jobs_df = job_system.scrape_new_jobs(keywords, pages_input)
            
            if not new_jobs_df.empty:
                st.success(f"✅ Scraped {len(new_jobs_df)} jobs!")
                
                # Identify truly new jobs
                truly_new_jobs = job_system.identify_new_jobs(new_jobs_df)
                
                if not truly_new_jobs.empty:
                    st.info(f"🆕 Found {len(truly_new_jobs)} new jobs!")
                    
                    # Show new jobs
                    st.subheader("New Jobs Found")
                    st.dataframe(truly_new_jobs, use_container_width=True)
                    
                    # Option to save new jobs
                    if st.button("💾 Add to Database"):
                        if os.path.exists(job_system.data_path):
                            existing_df = pd.read_csv(job_system.data_path)
                            updated_df = pd.concat([existing_df, truly_new_jobs], ignore_index=True)
                        else:
                            updated_df = truly_new_jobs
                        
                        updated_df.to_csv(job_system.data_path, index=False)
                        st.success("✅ New jobs added to database!")
                        st.info("🎯 Next: Go to **🤖 Train Clustering** to create job clusters!")
                
                else:
                    st.info("No new jobs found (all jobs already in database)")
                
            else:
                st.warning("No jobs found. Try different keywords or check the website.")
    
    elif page == "📧 Email Setup":
        st.header("📧 Email Configuration")
        
        st.info("Configure email settings to receive job alerts")
        
        email_enabled = st.checkbox(
            "Enable Email Alerts",
            value=job_system.config.get('email_enabled', False)
        )
        
        if email_enabled:
            sender_email = st.text_input(
                "Sender Gmail Address",
                value=job_system.config.get('sender_email', ''),
                help="Use a Gmail address with App Password enabled"
            )
            
            sender_password = st.text_input(
                "Gmail App Password",
                type="password",
                value=job_system.config.get('sender_password', ''),
                help="Generate an App Password in your Google Account settings"
            )
            
            recipient_email = st.text_input(
                "Recipient Email",
                value=job_system.config.get('recipient_email', ''),
                help="Email address to receive job alerts"
            )
            
            # Test email functionality
            if st.button("📧 Send Test Email"):
                if sender_email and sender_password and recipient_email:
                    try:
                        yag = yagmail.SMTP(sender_email, sender_password)
                        yag.send(
                            to=recipient_email,
                            subject="Test Email - Job Alert System",
                            contents="This is a test email from your Job Alert System. Setup successful! 🎉"
                        )
                        st.success("✅ Test email sent successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to send test email: {str(e)}")
                else:
                    st.warning("Please fill in all email fields")
        
        # Save email configuration
        if st.button("💾 Save Email Settings"):
            job_system.config.update({
                'email_enabled': email_enabled,
                'sender_email': sender_email if email_enabled else '',
                'sender_password': sender_password if email_enabled else '',
                'recipient_email': recipient_email if email_enabled else ''
            })
            job_system.save_config()
            st.success("✅ Email settings saved!")
    
    elif page == "📈 Analytics":
        st.header("📈 Job Market Analytics")
        
        if os.path.exists(job_system.data_path):
            df = pd.read_csv(job_system.data_path)
            
            # Check if clustering has been done
            if 'Cluster' in df.columns:
                # Cluster-based analytics
                st.subheader("🎯 Cluster Analysis")
                
                # Cluster distribution pie chart
                cluster_counts = df['Cluster'].value_counts().sort_index()
                fig_cluster = px.pie(
                    values=cluster_counts.values,
                    names=[f"Cluster {i}" for i in cluster_counts.index],
                    title="Job Distribution by Cluster"
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # Skills analysis by cluster
                st.subheader("🛠️ Skills Analysis by Cluster")
                selected_cluster = st.selectbox(
                    "Select Cluster for Skills Analysis:",
                    sorted(df['Cluster'].unique()),
                    format_func=lambda x: f"Cluster {x}"
                )
                
                cluster_data = df[df['Cluster'] == selected_cluster]
                all_skills = []
                for skills in cluster_data['Skills'].dropna():
                    all_skills.extend(custom_tokenizer(skills))
                
                if all_skills:
                    skill_counts = pd.Series(all_skills).value_counts().head(15)
                    fig_skills = px.bar(
                        x=skill_counts.values,
                        y=skill_counts.index,
                        orientation='h',
                        title=f"Top Skills in Cluster {selected_cluster}"
                    )
                    st.plotly_chart(fig_skills, use_container_width=True)
            
            # Experience level analysis
            st.subheader("💼 Experience Level Distribution")
            exp_counts = df['Experience'].value_counts()
            fig_exp = px.pie(
                values=exp_counts.values,
                names=exp_counts.index,
                title="Jobs by Experience Level"
            )
            st.plotly_chart(fig_exp, use_container_width=True)
            
            # Location analysis
            st.subheader("📍 Geographic Distribution")
            location_counts = df['Location'].value_counts().head(10)
            fig_loc = px.bar(
                x=location_counts.values,
                y=location_counts.index,
                orientation='h',
                title="Top 10 Job Locations"
            )
            st.plotly_chart(fig_loc, use_container_width=True)
            
            # Overall skills analysis
            st.subheader("🛠️ Most In-Demand Skills (Overall)")
            all_skills = []
            for skills in df['Skills'].dropna():
                all_skills.extend(custom_tokenizer(skills))
            
            if all_skills:
                skill_counts = pd.Series(all_skills).value_counts().head(20)
                fig_skills = px.bar(
                    x=skill_counts.values,
                    y=skill_counts.index,
                    orientation='h',
                    title="Top 20 Skills Mentioned"
                )
                st.plotly_chart(fig_skills, use_container_width=True)
        
        else:
            st.warning("No data available for analytics. Please load or scrape job data first.")
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Status")
    
    model_status = "✅ Loaded" if job_system.kmeans_model else "❌ Not Found"
    st.sidebar.markdown(f"**Models:** {model_status}")
    
    email_status = "✅ Configured" if job_system.config.get('email_enabled') else "❌ Not Setup"
    st.sidebar.markdown(f"**Email:** {email_status}")
    
    # Data status
    if os.path.exists(job_system.data_path):
        df = pd.read_csv(job_system.data_path)
        data_status = f"✅ {len(df)} jobs"
        if 'Cluster' in df.columns:
            data_status += f" ({df['Cluster'].nunique()} clusters)"
    else:
        data_status = "❌ No data"
    st.sidebar.markdown(f"**Data:** {data_status}")
    
    if job_system.job_history.get('last_check'):
        last_check = datetime.fromisoformat(job_system.job_history['last_check'])
        time_diff = datetime.now() - last_check
        if time_diff.days > 1:
            check_status = f"⚠️ {time_diff.days} days ago"
        else:
            check_status = f"✅ {time_diff.seconds//3600}h ago"
        st.sidebar.markdown(f"**Last Check:** {check_status}")

if __name__ == "__main__":
    main()
