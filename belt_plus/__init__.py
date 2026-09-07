# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

from .bert import BertBase
from .bert_with_pooling import BertWithPooling, ChunkedDataset
from .bert_truncated import BertTruncated, TruncatedDataset
from .bert_classifier_with_pooling import BertClassifierWithPooling
from .bert_classifier_truncated import BertClassifierTruncated
from .exceptions import InconsistentSplittingParamsException
from .tensor_utils import list_of_tensors_deep_equal, split_overlapping

from .neural_networks.pooling_nn import PoolingNeuralNetwork
from .neural_networks.truncation_nn import TruncationNeuralNetwork
from .neural_networks.base_nn import BaseNeuralNetwork

__all__ = [
    "BertBase",
    "BertWithPooling",
    "ChunkedDataset",
    "BertTruncated",
    "TruncatedDataset",
    "BertClassifierWithPooling",
    "BertClassifierTruncated",
    "InconsistentSplittingParamsException",
    "list_of_tensors_deep_equal",
    "split_overlapping",
    "PoolingNeuralNetwork",
    "TruncationNeuralNetwork",
    "BaseNeuralNetwork",
]