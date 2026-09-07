# Copyright (c) 2026 Paul van Rijen, Patrick Ruch
# HES-SO Geneva / SIB Swiss Institute of Bioinformatics
# Licensed under the MIT License

"""fulltext-diagnostic: Diagnose and transfer full-text knowledge for biomedical screening.

This package implements the diagnostic protocol described in:

    van Rijen, P. and Ruch, P. (2026). "Beyond abstracts: diagnosing full-text
    effects in biomedical screening." Database (Oxford University Press).

The protocol decomposes the effect of full-text processing into a Training
Effect (TE) and an Inference Effect (IE). When these cancel (TE < 0, IE > 0),
Privileged Features Distillation (PFD) transfers the full-text signal to an
abstract-only student model at zero additional inference cost.
"""

__version__ = "0.1.0"
