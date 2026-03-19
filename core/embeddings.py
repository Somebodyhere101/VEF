"""
Token embeddings with IDF weighting.

Meaning is captured through co-occurrence: words appearing in similar
contexts get similar embeddings. IDF ensures content words dominate
over structural words ("gravity" matters more than "the").
"""
import functools
import numpy as np
import os
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Embeddings:

    def __init__(self, data_dir):
        E = np.load(os.path.join(data_dir, 'token_embeds.npy')).astype(np.float32)
        self.vocab_size, self.dim = E.shape
        self.raw = E

        norms = np.linalg.norm(E, axis=1, keepdims=True)
        self.normed = E / np.maximum(norms, 1e-10)

        self.idf = np.load(os.path.join(data_dir, 'idf_weights.npy')).astype(np.float32)

        # GPU copies
        self.E = torch.tensor(E, dtype=torch.float32, device=DEVICE)
        self.E_normed = torch.tensor(self.normed, dtype=torch.float32, device=DEVICE)
        self.idf_t = torch.tensor(self.idf, dtype=torch.float32, device=DEVICE)

    def embed(self, text, tokenizer, attention=None):
        """IDF-weighted text embedding, optionally contextualized by attention.

        Without attention: static bag-of-words (fast, used for corpus embeddings).
        With attention: contextual — word order and relationships matter.
        "bank near river" gets a different embedding than "bank near money."
        """
        # LRU cache for static embeddings (no attention) — the #1 perf optimization
        if attention is None:
            return self._embed_static_cached(text, tokenizer)

        ids = [t for t in tokenizer.encode(text).ids if 0 <= t < self.vocab_size]
        if not ids:
            return None

        # Contextual: run attention, then IDF-weighted pool
        if len(ids) >= 3:
            ctx = attention.contextualize(ids, self.E, causal=False)
            # IDF-weighted pool over contextualized representations
            weights = self.idf_t[torch.tensor(ids, device=ctx.device)]
            total = weights.sum()
            if total < 1e-10:
                return None
            emb = (ctx * weights.unsqueeze(1)).sum(0) / total
            emb = emb / (emb.norm() + 1e-10)
            return emb.cpu().numpy().astype(np.float32)

        # Fewer than 3 tokens with attention — fall back to static
        return self._embed_static_cached(text, tokenizer)

    @functools.lru_cache(maxsize=4096)
    def _embed_static_cached(self, text, tokenizer):
        """Cached static IDF-weighted embedding."""
        ids = [t for t in tokenizer.encode(text).ids if 0 <= t < self.vocab_size]
        if not ids:
            return None

        # Vectorized static embedding
        ids_arr = np.array(ids)
        weights = self.idf[ids_arr]
        total_weight = weights.sum()
        if total_weight < 1e-10:
            return None
        emb = (self.raw[ids_arr] * weights[:, None]).sum(axis=0).astype(np.float64)
        emb /= total_weight
        norm = np.linalg.norm(emb)
        if norm < 1e-10:
            return None
        return (emb / norm).astype(np.float32)
