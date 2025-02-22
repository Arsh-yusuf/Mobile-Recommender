# Mobile Recommendation System

## Overview
This project implements a **mobile recommendation system** using **DistilBERT** for semantic understanding of user queries. The system takes a text description as input (e.g., "Oppo phone with 8GB RAM, 128GB ROM, 5000mAh battery") and recommends the most relevant mobile phones based on embeddings and cosine similarity.

## Features
- Uses **DistilBERT (distilbert-base-uncased)** for text embeddings.
- Extracts numerical values like **RAM, Storage, Battery, and Price** from user queries.
- Filters recommendations based on extracted specifications.
- Computes **cosine similarity** between the user query and stored mobile embeddings.
- Returns the top **K recommended mobile phones**.

## Dataset
The dataset contains:
- **MobileName**: Name of the mobile phone.
- **Description**: Text description including specifications.
- **Price**: Price of the phone.
- **Embedding**: DistilBERT-generated vector representation of the description.

## Model Used
- **Transformer Model**: `distilbert-base-uncased`
- **Library**: `transformers`
- **Embedding Generation**:
  - Tokenization using `DistilBertTokenizer.from_pretrained("distilbert-base-uncased")`
  - Feature extraction using `DistilBertModel.from_pretrained("distilbert-base-uncased")`

## Installation
To run the project, install the required dependencies:
pip install torch transformers pandas scikit-learn numpy

## Conclusion
This project provides an efficient recommendation system using **transformers** to understand natural language queries and match them with mobile phone specifications.

