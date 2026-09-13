
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out sentence_transformers BEFORE any rag.* module is imported.
# This prevents SentenceTransformer from loading model weights from disk
# (which hangs / takes minutes) during the test collection phase.
# ---------------------------------------------------------------------------
_fake_st_model = MagicMock()
_fake_st_model.encode.return_value = [[0.0] * 384]

_st_module = MagicMock()
_st_module.SentenceTransformer.return_value = _fake_st_model
sys.modules.setdefault("sentence_transformers", _st_module)

# If rag.embeddings was already imported (e.g. by a previous test session
# that cached .pyc files), replace its singleton too.
if "rag.embeddings" in sys.modules:
    sys.modules["rag.embeddings"].model = _fake_st_model
