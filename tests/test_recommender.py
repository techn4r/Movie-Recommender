import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

class TestMovieRecommenderUnit:
    def test_predict_without_fit_raises(self):
        from src.recommender import MovieRecommender
        recommender = MovieRecommender(model='svd')
        with pytest.raises(ValueError, match="Model not fitted"):
            recommender.recommend_for_user(user_id=1)

class TestFilterColdStart:
    def test_filter_removes_sparse_users(self):
        from src.preprocessing import filter_cold_start
        ratings = pd.DataFrame({
            'user_id': [1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3],
            'movie_id': [1, 2, 3, 4, 5, 1, 1, 2, 3, 4, 5],
            'rating': [5.0] * 11
        })
        filtered = filter_cold_start(ratings, min_u=3, min_i=1)
        assert 2 not in filtered['user_id'].values
