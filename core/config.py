"""
VEF Configuration — all tunable thresholds in one place.

These are manually calibrated constants, not gradient-trained parameters.
Adjust them to tune behavior for different corpora or domains.
"""


class Config:
    """Central configuration for all VEF thresholds and hyperparameters."""

    # Awareness thresholds (calibrated from known/unknown word distributions)
    AWARENESS_STRONG_ENERGY = 0.95
    AWARENESS_STRONG_CONC = 0.30
    AWARENESS_MEANING_ENERGY = 0.75
    AWARENESS_MEANING_CONC = 0.23
    AWARENESS_NOISE_ENERGY = 0.72
    AWARENESS_NOISE_CONC = 0.30

    # Retrieval scoring weights
    RETRIEVAL_QQ_BASE = 0.10
    RETRIEVAL_QQ_CONF_SCALE = 0.70
    RETRIEVAL_RB_BASE = 0.85
    RETRIEVAL_RB_CONF_SCALE = 0.60
    RETRIEVAL_RS_BASE = 0.05
    RETRIEVAL_RS_CONF_SCALE = 0.10

    # Quality penalties
    RETRIEVAL_CODE_PENALTY = 0.15
    RETRIEVAL_CODE_DISCUSSION_PENALTY = 0.2
    RETRIEVAL_LONG_RESPONSE_PENALTY = 0.6

    # Confidence thresholds
    CONFIDENCE_ACCEPT = 0.3
    REFINEMENT_CONVERGENCE = 0.95

    # Lateral inhibition
    LATERAL_AMPLIFY = 0.15
    LATERAL_SUPPRESS = 0.2
    LATERAL_FLOOR = 0.4

    # System prompt injection
    SYSTEM_INJECT_HIGH = 0.9
    SYSTEM_INJECT_LOW = 0.8

    # Retrieval short-query threshold
    SHORT_QUERY_WORDS = 3


# Global default instance
DEFAULT = Config()
