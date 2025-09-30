from src.data_loader import load_data
from src.collaborative_filtering import build_cf_model, recommend_cf
from src.hybrid_recommender import hybrid_recommend
from src.evaluation import precision_at_k, recall_at_k

def run_pipeline():
    # Load data
    df = load_data()

    # Train collaborative filtering
    model, dataset = build_cf_model(df)

    # Recommend for user 1
    cf_recs = recommend_cf(model, dataset, user_id=1, top_n=5)
    content_recs = ["GOOG", "MSFT", "NVDA", "TSLA", "AAPL"]  # mock results
    hybrid_recs = hybrid_recommend(cf_recs, content_recs)

    print("Collaborative:", cf_recs)
    print("Hybrid:", hybrid_recs)

    # Evaluate (mock actuals for demo)
    actual = ["AAPL", "TSLA"]
    print("Precision@5:", precision_at_k(actual, hybrid_recs))
    print("Recall@5:", recall_at_k(actual, hybrid_recs))

if __name__ == "__main__":
    run_pipeline()
