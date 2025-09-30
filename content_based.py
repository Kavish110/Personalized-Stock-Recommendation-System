import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def build_content_model(stock_metadata):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(stock_metadata["description"])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

def recommend_content(stock_metadata, stock_id, similarity_matrix, top_n=5):
    stock_index = stock_metadata[stock_metadata["stock_id"] == stock_id].index[0]
    sim_scores = list(enumerate(similarity_matrix[stock_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_stocks = [stock_metadata.iloc[i[0]]["stock_id"] for i in sim_scores[1:top_n+1]]
    return top_stocks
