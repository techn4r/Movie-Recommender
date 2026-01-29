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
    return matrix

class TestHybridRecommender:
    def test_fit(self, sample_ratings_matrix):
        hybrid = HybridRecommender()
        hybrid.add_model('svd', SVDModel(n_factors=10, n_epochs=5), weight=0.6)
        hybrid.fit(sample_ratings_matrix, verbose=False)
        
        # Проверяем, что вложенная модель обучилась
        assert hybrid.models['svd'].user_factors is not None
