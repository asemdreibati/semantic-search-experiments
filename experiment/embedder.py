"""multilingual-e5-large (1024-d, ONNX) with mean pooling + L2 norm.

e5 was trained with role prefixes: queries MUST start with "query: " and
documents with "passage: ". Dropping them compresses every cosine into a
narrow band, which makes a relevance threshold impossible to set.

Model files: set E5_MODEL_DIR (default ./models/fast-multilingual-e5-large),
downloadable from
https://storage.googleapis.com/qdrant-fastembed/fast-multilingual-e5-large.tar.gz
"""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(os.environ.get(
    "E5_MODEL_DIR",
    Path(__file__).resolve().parent / "models" / "fast-multilingual-e5-large",
))
MAX_TOKENS = 512


class E5:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.tok.enable_padding()
        self.sess = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def _embed(self, texts, batch_size=16):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[i:i + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            vec = (hidden * m).sum(1) / m.sum(1)
            out.append(vec / np.linalg.norm(vec, axis=1, keepdims=True))
        return np.vstack(out).astype(np.float32)

    def queries(self, texts, prefix=True):
        return self._embed([("query: " if prefix else "") + t for t in texts])

    def passages(self, texts, prefix=True):
        return self._embed([("passage: " if prefix else "") + t for t in texts])


class BGEM3:
    """BAAI/bge-m3 dense embeddings - the model the production system uses
    (identified from the Nuxeo query vector, see identify_model.md).
    bge-m3 takes no query/passage prefixes; `prefix` is accepted and ignored."""

    def __init__(self, model_id=os.environ.get("BGE_M3_DIR", "BAAI/bge-m3")):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_id, device="cpu")

    def _embed(self, texts):
        return self.model.encode(texts, batch_size=8, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)

    def queries(self, texts, prefix=True):
        return self._embed(texts)

    def passages(self, texts, prefix=True):
        return self._embed(texts)


# variant name -> (model, prefix chunks with doc title, use e5 role prefixes)
VARIANTS = {
    "title_prefix": ("e5", True, True),        # e5 recommended setup
    "no_title": ("e5", False, True),           # ablation
    "no_e5_prefix": ("e5", True, False),       # ablation
    "bge_m3": ("bge-m3", True, False),         # production model
    "bge_m3_no_title": ("bge-m3", False, False),
}
_MODELS = {"e5": E5, "bge-m3": BGEM3}
_loaded = {}


def embedder_for(variant):
    model, title, prefix = VARIANTS[variant]
    if model not in _loaded:
        _loaded[model] = _MODELS[model]()
    return _loaded[model], title, prefix
