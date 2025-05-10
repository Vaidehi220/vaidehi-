import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load saved models and data
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
svd_model = joblib.load('svd_model.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')
df = pd.read_csv('cleaned_data.csv')

# Transform product descriptions
tfidf_matrix = tfidf_vectorizer.transform(df['Description'])

# Streamlit interface
st.title("🛒 Product Recommendation System")
product_input = st.text_input("Enter a product name (e.g., 'white metal lantern'):")

if st.button("Recommend"):
    product_input = product_input.lower()
    if product_input in df['Description'].values:
        idx = df[df['Description'] == product_input].index[0]
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        sim_indices = sim_scores.argsort()[::-1][1:6]

        st.subheader(f"🔍 Top 5 similar products to: **{product_input}**")
        for i in sim_indices:
            st.write(f"👉 {df.iloc[i]['Description']}")
    else:
        st.error("❌ Product not found.")
