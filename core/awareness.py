"""
Awareness — basis vs anti-basis knowledge detection.

The basis (128 SVD dimensions) captures what the model KNOWS.
The anti-basis (null space) captures what it DOESN'T.

For any input:
  - High basis energy, low concentration → REAL CONCEPT (model knows this)
  - Low basis energy, high concentration → NOISE (model has no information)
  - Medium energy, medium concentration → PARTIAL (model has some knowledge)

The geometry of the learned space tells the model where its knowledge ends.
"""
import numpy as np

from core.config import DEFAULT as CFG


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

        E = self.embeddings.raw
        idf = self.embeddings.idf

        token_embs = E[ids]
        weights = np.array([idf[t] for t in ids])
        total_w = weights.sum()
        if total_w < 1e-10:
            return 'noise', 0.0, 1.0, "no IDF weight"

        w = weights / total_w
        emb = (token_embs * w[:, None]).sum(0)

        # BASIS ENERGY: norm of projection onto learned space
        energy = float(np.linalg.norm(emb))

        # ANTI-BASIS SIGNAL: entropy of energy distribution across dimensions
        # Real concepts spread across many dimensions; noise concentrates in few
        concentration = 0.0
        if energy > 0:
            normalized = (emb / energy) ** 2
            entropy = -np.sum(normalized * np.log(normalized + 1e-10))
            max_entropy = np.log(len(emb))
            concentration = 1.0 - entropy / max_entropy

        # Classification thresholds calibrated from observed separation between
        # real concepts (energy ~0.8-1.0, conc ~0.14-0.20) and
        # gibberish (energy ~0.55-0.69, conc ~0.26-0.35)
        if energy > CFG.AWARENESS_STRONG_ENERGY and concentration < CFG.AWARENESS_STRONG_CONC:
            return 'partial', energy, concentration, "strong energy, slightly concentrated"
        elif energy > CFG.AWARENESS_MEANING_ENERGY and concentration < CFG.AWARENESS_MEANING_CONC:
            return 'meaning', energy, concentration, "strong basis projection, distributed energy"
        elif energy < CFG.AWARENESS_NOISE_ENERGY or concentration > CFG.AWARENESS_NOISE_CONC:
            return 'noise', energy, concentration, "weak basis / concentrated energy"
        else:
            return 'partial', energy, concentration, "borderline signal"
