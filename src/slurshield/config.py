# Backbone is provisional; Task 3 (backbone smoke test) confirms or flips it to the fallback.
BACKBONE = "jhu-clsp/mmbert-small"
FALLBACK_BACKBONE = "distilbert-base-multilingual-cased"
MAX_LENGTH = 48
CLASS_NAMES = ["clean", "explicit", "implicit", "action"]  # index order is fixed
BLOCK_LABELS = {"explicit", "implicit"}  # chat labels that block; clean/action allow
