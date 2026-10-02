
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out sentence_transformers BEFORE any rag.* module is imported.
# This prevents SentenceTransformer from loading model weights from disk
# (which hangs / takes minutes) during the test collection phase.
# ---------------------------------------------------------------------------
_fake_st_model = MagicMock()
def _mock_encode(texts, *args, **kwargs):
    def _vec(t: str) -> list[float]:
        v = [0.01] * 384
        for w in t.lower().split():
            idx = sum(ord(c) for c in w) % 384
            v[idx] += 1.0
        norm = (sum(x * x for x in v)) ** 0.5
        return [x / norm for x in v] if norm > 0 else v

    if isinstance(texts, str):
        return _vec(texts)
    return [_vec(t) for t in texts]

_fake_st_model.encode.side_effect = _mock_encode

_st_module = MagicMock()
_st_module.SentenceTransformer.return_value = _fake_st_model
sys.modules.setdefault("sentence_transformers", _st_module)

# If rag.embeddings was already imported (e.g. by a previous test session
# that cached .pyc files), replace its singleton too.
if "rag.embeddings" in sys.modules:
    sys.modules["rag.embeddings"].model = _fake_st_model
