# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

class InconsistentSplittingParamsException(Exception):
    pass
