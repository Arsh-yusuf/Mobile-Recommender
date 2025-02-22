import streamlit as st
import pandas as pd
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load dataset with embeddings
df = pd.read_pickle("mobile_embeddings.pkl")

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_embedding(text):
    """Generate DistilBERT embedding for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def extract_numeric_value(query, keyword):
    """Extract a numeric value (like RAM, storage, battery, price) from the user query."""
    match = re.search(rf"(\d+)\s*{keyword}", query, re.IGNORECASE)
    return int(match.group(1)) if match else None

def extract_brand(query, df):
    """Extract brand name from user query based on dataset brands."""
    brands = set(df['MobileName'].str.split().str[0].str.lower().unique())  # Convert brands to lowercase
    for word in query.lower().split():
        if word in brands:
            return word.capitalize()
    return None

def recommend_mobiles(user_query, top_k=5):
    """Find the most similar mobiles based on user query with full column filtering."""
    query_embedding = get_embedding(user_query).reshape(1, -1)

    # Extract filters
    required_ram = extract_numeric_value(user_query, "GB RAM")
    required_storage = extract_numeric_value(user_query, "GB ROM")
    required_battery = extract_numeric_value(user_query, "mAh")
    max_price = extract_numeric_value(user_query, "less than")
    required_brand = extract_brand(user_query, df)

    # Extract RAM and storage from dataset
    df['RAM'] = df['Description'].apply(lambda x: extract_numeric_value(x, "GB RAM"))
    df['Storage'] = df['Description'].apply(lambda x: extract_numeric_value(x, "GB ROM"))
    df['Battery'] = df['Description'].apply(lambda x: extract_numeric_value(x, "mAh"))

    # Apply filters
    df_filtered = df.copy()
    if required_ram:
        df_filtered = df_filtered[df_filtered['RAM'] == required_ram]
    if required_storage:
        df_filtered = df_filtered[df_filtered['Storage'] == required_storage]
    if required_battery:
        df_filtered = df_filtered[df_filtered['Battery'] >= required_battery]
    if max_price:
        df_filtered = df_filtered[df_filtered['Price'] <= max_price]
    if required_brand:
        df_filtered = df_filtered[df_filtered['MobileName'].str.lower().str.startswith(required_brand.lower())]

    if df_filtered.empty:
        return "No matching mobiles found. Try a different query."

    # Convert stored embeddings to numpy array
    mobile_embeddings = np.vstack(df_filtered['Embedding'].values)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, mobile_embeddings)[0]

    # Get top K recommendations
    top_indices = similarities.argsort()[-top_k:][::-1]
    recommendations = df_filtered.iloc[top_indices][['MobileName', 'Price', 'Description']]

    return recommendations

# Streamlit UI
st.title("📱 Mobile Recommendation System")
st.write("Enter your mobile preference, and we'll find the best options for you!")

# User input
txt_input = st.text_input("Describe your ideal phone (e.g., 'Redmi, 8GB RAM, 128GB ROM, under 20000'):")

if txt_input:
    st.subheader("🔍 Recommended Mobiles:")
    results = recommend_mobiles(txt_input)
    
    if isinstance(results, str):
        st.write(results)
    else:
        st.dataframe(results)
