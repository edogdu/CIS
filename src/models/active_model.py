"""
Current production model selection.
Switch models here without changing the rest of the codebase.
"""

from models.classifiers.hetero_classifier import GNNHeteroClassifierModel

ActiveModel = GNNHeteroClassifierModel
