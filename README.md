# Personalized-Stock-Recommendation-System


This project implements a **hybrid stock recommendation engine** that combines **collaborative filtering, content-based filtering, and learning-to-rank**.  
It is designed to mimic real-world personalization systems, to help users discover the most relevant financial assets.  

### 🔹 Workflow Diagram  

```mermaid
flowchart TD
    A[📂 Data Loader] --> B[📊 EDA & Insights]
    B --> C1[🤝 Collaborative Filtering]
    B --> C2[📑 Content-Based Filtering]
    C1 --> D[⚡ Hybrid Recommender]
    C2 --> D[⚡ Hybrid Recommender]
    D --> E[📈 Ranker: Diversification & Risk Metrics]
    E --> F[✅ Final Recommendations]

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
```
---

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/personalized-stock-recommender.git
cd personalized-stock-recommender
pip install -r requirements.txt
```
