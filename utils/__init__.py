"""
Utility modules for data processing, training, evaluation, and visualization.
"""

# Import non-torch dependent modules
from .data_utils import StockDataLoader, get_default_split_dates
from .evaluation import BaselineEvaluator, RegressionMetrics, ClassificationMetrics, DirectionalAccuracy

# Try to import torch-dependent modules
try:
    from .dataset import StockSequenceDataset, save_scalers, load_scalers
    from .visualization import StockVisualizer
    from .trainer import Trainer
    _TORCH_AVAILABLE = True
except ImportError:
    StockSequenceDataset = None
    save_scalers = None
    load_scalers = None
    StockVisualizer = None
    Trainer = None
    _TORCH_AVAILABLE = False

__all__ = [
    'StockSequenceDataset',
    'save_scalers',
    'load_scalers',
    'StockDataLoader',
    'get_default_split_dates',
    'BaselineEvaluator',
    'RegressionMetrics',
    'ClassificationMetrics',
    'DirectionalAccuracy',
    'StockVisualizer',
    'Trainer',
]

