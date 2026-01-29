import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    UserBasedCF, ItemBasedCF, SVDModel, 
    ContentBasedFilter, HybridRecommender
)

@pytest.fixture
def sample_ratings_matrix():
    np.random.seed(42)
    matrix = np.random.randint(0, 6, (50, 30)).astype(float)
    matrix[:10, :5] = np.random.randint(4, 6, (10, 5))
    return matrix

class TestSVDModel:
    def test_fit(self, sample_ratings_matrix):
        model = SVDModel(n_factors=10, n_epochs=5)
        model.fit(sample_ratings_matrix, verbose=False)
        assert model.user_factors is not None
        assert model.item_factors is not None

class TestHybridRecommender:
    def test_add_model(self):
        hybrid = HybridRecommender()
        hybrid.add_model('svd', SVDModel(n_factors=10), weight=0.6)
        assert 'svd' in hybrid.models

    def test_fit(self, sample_ratings_matrix):
        """Исправлено: корректная проверка обучения моделей"""
        hybrid = HybridRecommender()
        hybrid.add_model('svd', SVDModel(n_factors=10, n_epochs=5), weight=0.6)
        hybrid.add_model('cf', ItemBasedCF(k_neighbors=10), weight=0.4)
        
        hybrid.fit(sample_ratings_matrix, verbose=False)
        assert hybrid.models['svd'].user_factors is not None
        assert hybrid.models['cf'].item_similarity is not None
