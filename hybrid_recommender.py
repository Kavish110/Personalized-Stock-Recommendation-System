def hybrid_recommend(cf_recs, content_recs, alpha=0.6):
    """
    alpha = weight for collaborative filtering
    (1 - alpha) = weight for content-based
    """
    scores = {}
    for i, stock in enumerate(cf_recs):
        scores[stock] = scores.get(stock, 0) + alpha * (len(cf_recs) - i)

    for i, stock in enumerate(content_recs):
        scores[stock] = scores.get(stock, 0) + (1 - alpha) * (len(content_recs) - i)

    return sorted(scores, key=scores.get, reverse=True)
