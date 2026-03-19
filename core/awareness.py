"""
Awareness — the war between basis and anti-basis.

The basis (128 SVD dimensions) captures what the model KNOWS.
The anti-basis (null space) captures what it DOESN'T.

For any input:
  - High basis energy, low concentration → REAL CONCEPT (model knows this)
  - Low basis energy, high concentration → NOISE (model has no information)
  - Medium energy, medium concentration → PARTIAL (model has some knowledge)

This is not a threshold. It's the GEOMETRY of the learned space
telling the model where its knowledge ends.

The basis and anti-basis are two sides of the same coin:
one enables generalization, the other detects its limits.
"""
import numpy as np


class Awareness:
    """Detect what the model knows vs doesn't know from embedding geometry."""

    def __init__(self, embeddings, tokenizer):
        self.embeddings = embeddings
        self.tokenizer = tokenizer

    def measure(self, text):
        """Measure how much of this text falls within the model's knowledge.

        Returns: (category, energy, concentration, detail)
          category: 'meaning' | 'partial' | 'noise'
          energy: how strongly the text projects onto the learned basis (0-1)
          concentration: how spread the energy is (0=spread=real, 1=concentrated=noise)
        """
        ids = [t for t in self.tokenizer.encode(text).ids
               if 0 <= t < self.embeddings.vocab_size]
        if not ids:
            return 'noise', 0.0, 1.0, "empty input"

        # Token embeddings in the basis (128d SVD space)
        E = self.embeddings.raw
        idf = self.embeddings.idf

        token_embs = E[ids]
        weights = np.array([idf[t] for t in ids])
        total_w = weights.sum()
        if total_w < 1e-10:
            return 'noise', 0.0, 1.0, "no IDF weight"

        # Weighted average embedding in basis space
        w = weights / total_w
        emb = (token_embs * w[:, None]).sum(0)

        # BASIS ENERGY: norm of projection onto learned space
        energy = float(np.linalg.norm(emb))

        # ANTI-BASIS SIGNAL: how concentrated is the energy?
        # Real concepts spread across many dimensions (rich connections)
        # Noise concentrates in few random dimensions (no structure)
        concentration = 0.0
        if energy > 0:
            normalized = (emb / energy) ** 2
            entropy = -np.sum(normalized * np.log(normalized + 1e-10))
            max_entropy = np.log(len(emb))
            concentration = 1.0 - entropy / max_entropy

        # Classify by the war between basis and anti-basis
        # Thresholds calibrated from the observed separation between
        # real concepts (energy ~0.8-1.0, conc ~0.14-0.20) and
        # gibberish (energy ~0.55-0.69, conc ~0.26-0.35)
        if energy > 0.75 and concentration < 0.23:
            return 'meaning', energy, concentration, "strong basis projection, distributed energy"
        elif energy > 0.95 and concentration < 0.30:
            # Very high energy = word IS in vocabulary, even if concentrated
            return 'partial', energy, concentration, "strong energy, slightly concentrated"
        elif energy < 0.72 or concentration > 0.30:
            return 'noise', energy, concentration, "weak basis / concentrated energy"
        else:
            return 'partial', energy, concentration, "borderline signal"
