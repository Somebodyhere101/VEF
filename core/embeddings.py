"""
Token embeddings with IDF weighting.

Meaning is captured through co-occurrence: words appearing in similar
contexts get similar embeddings. IDF ensures content words dominate
over structural words ("gravity" matters more than "the").
"""
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
        ids = [t for t in tokenizer.encode(text).ids if 0 <= t < self.vocab_size]
        if not ids:
            return None

        # Contextual: run attention, then IDF-weighted pool
        if attention is not None and len(ids) >= 3:
            ctx = attention.contextualize(ids, self.E, causal=False)
            # IDF-weighted pool over contextualized representations
            weights = self.idf_t[torch.tensor(ids, device=ctx.device)]
            total = weights.sum()
            if total < 1e-10:
                return None
            emb = (ctx * weights.unsqueeze(1)).sum(0) / total
            emb = emb / (emb.norm() + 1e-10)
            return emb.cpu().numpy().astype(np.float32)

        # Static: IDF-weighted average (fast)
        emb = np.zeros(self.dim, dtype=np.float64)
        total_weight = 0.0
        for tid in ids:
            w = float(self.idf[tid])
            emb += w * self.raw[tid]
            total_weight += w

        if total_weight < 1e-10:
            return None
        emb /= total_weight
        norm = np.linalg.norm(emb)
        if norm < 1e-10:
            return None
        return (emb / norm).astype(np.float32)
