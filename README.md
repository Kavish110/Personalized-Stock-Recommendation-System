# Personalized-Stock-Recommendation-System


This project implements a **hybrid stock recommendation engine** that combines **collaborative filtering, content-based filtering, and learning-to-rank**.  
It is designed to mimic real-world personalization systems, like those used at Robinhood, to help users discover the most relevant financial assets.  

---

## 🚀 Features
- **Collaborative Filtering** (LightFM) for user–stock interactions  
- **Content-Based Filtering** (TF-IDF + cosine similarity on stock metadata)  
- **Hybrid Recommender** that blends multiple approaches  
- **Learning-to-Rank** with XGBoost for personalized ranking  
- **Evaluation Module** with precision@k, recall@k, and A/B test simulation  

---

## 📂 Project Structure

```bash
personalized-stock-recommender/
│── data/
│   └── sample_portfolios.csv
│
│── notebooks/
│   └── EDA.ipynb
│
│── src/
│   ├── data_loader.py
│   ├── collaborative_filtering.py
│   ├── content_based.py
│   ├── hybrid_recommender.py
│   ├── ranker.py
│   ├── evaluation.py
│
│── main.py
│── requirements.txt
│── README.md
'''
---

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/personalized-stock-recommender.git
cd personalized-stock-recommender
pip install -r requirements.txt
