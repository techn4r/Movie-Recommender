import pytest
import numpy as np
import sys
from pathlib import Path

# Путь для корректного импорта из src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    rmse, mae, precision_at_k, recall_at_k, f1_at_k,
    ndcg_at_k, mean_reciprocal_rank, hit_rate_at_k,
    coverage, diversity, novelty
)

class TestRatingMetrics:
    def test_rmse_calculation(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        assert rmse(y_true, y_pred) == 1.0

class TestRankingMetrics:
    def test_precision_at_k_partial(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5, 7, 9}
        assert precision_at_k(recommended, relevant, 5) == 0.6

class TestEdgeCases:
    def test_k_larger_than_list(self):
        """Проверка логики при k > длины списка"""
        recommended = [1, 2, 3]
        relevant = {1, 2, 3, 4, 5}
        # 3 релевантных / k (10) = 0.3
        p = precision_at_k(recommended, relevant, 10)
        assert p == 0.3
