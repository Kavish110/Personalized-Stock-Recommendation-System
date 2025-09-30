import numpy as np
from sklearn.metrics import precision_score, recall_score

def precision_at_k(actual, predicted, k=5):
    pred_k = predicted[:k]
    return len(set(pred_k) & set(actual)) / k

def recall_at_k(actual, predicted, k=5):
    pred_k = predicted[:k]
    return len(set(pred_k) & set(actual)) / len(actual) if actual else 0

def simulate_ab_test(metrics_a, metrics_b):
    lift = np.mean(metrics_b) - np.mean(metrics_a)
    return {"lift": lift, "better": "B" if lift > 0 else "A"}
