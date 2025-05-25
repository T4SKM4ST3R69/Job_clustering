# Job_clustering_using_Kmeans

This project is a simple web application built with Streamlit that demonstrates **KMeans clustering** on text data. It uses `KMeans-clustering` and a `TfidfVectorizer` to group text into clusters and display the results interactively.

## Features

- Upload text and get cluster predictions
- KMeans clustering
- TF-IDF vectorization of input text
- Clean and interactive UI with Streamlit

## Project Structure

```
.
├── Kmeans.ipynb               # Notebook for training and experimenting with KMeans
├── streamlit_apppy            # Streamlit app for the frontend
├── kmeans_model.joblib        # Trained KMeans model
├── tfidf_vectorizer.joblib    # Trained TF-IDF vectorizer
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/T4SKM4ST3R69/Job_clustering
cd kmeans-clustering-app
```

### 2. Install dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run streamlit_app.py
```

##  Training (Optional)

You can retrain the model using `Kmeans.ipynb`, which includes data preprocessing, vectorization with TF-IDF, and fitting a KMeans clustering model.

##  Requirements

See `requirements.txt` for a list of dependencies, which includes:

