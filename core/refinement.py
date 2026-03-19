"""
Iterative Refinement — recursive embedding convergence.

A loop that:
  1. Embeds the current query
  2. Retrieves the closest corpus entry
  3. Blends the query embedding with the retrieved embedding
  4. Repeats until the embedding stops changing (convergence)

The number of steps is not fixed — it emerges from the data.
Convergence indicates the model has focused on a stable interpretation.
"""
import numpy as np
import torch

from core.config import DEFAULT as CFG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Refinement:
    """Iterative refinement loop via embedding convergence."""

    def __init__(self, embeddings, corpus, tokenizer):
        self.embeddings = embeddings
        self.corpus = corpus
        self.tokenizer = tokenizer

    def refine(self, query, max_steps=8):
        """Iteratively refine a query by embedding-space convergence.

        Each step:
          1. Embed the current thought
          2. Retrieve the nearest corpus entry
          3. Blend the embedding toward the retrieved entry
          4. Check convergence (cosine similarity > 0.95)

        Returns: (final_result, trace, n_steps)
        """
        thought = query
        trace = []
        prev_emb = None
        step = 0

        for step in range(max_steps):
            emb = self.embeddings.embed(thought, self.tokenizer)
            if emb is None:
                break

            if prev_emb is not None:
                similarity = float(np.dot(emb, prev_emb))
                trace.append(f"Step {step}: similarity = {similarity:.4f}")
                if similarity > CFG.REFINEMENT_CONVERGENCE:
                    trace.append(f"Converged at step {step}")
                    break
            prev_emb = emb.copy()

            q_t = torch.tensor(emb, dtype=torch.float32, device=DEVICE)
            scores = self.corpus.q_embeds @ q_t
            top_score, top_idx = scores.topk(1)
            idx = top_idx[0].item()

            if idx >= len(self.corpus.entries):
                break

            match = self.corpus.extract_response(self.corpus.entries[idx])
            if not match or len(match) < 10:
                break

            trace.append(f"Step {step}: \"{thought[:50]}\" -> \"{match[:50]}\"")

            match_emb = self.embeddings.embed(match[:200], self.tokenizer)
            if match_emb is None:
                break

            alpha = (step + 1) / max_steps
            refined = (1 - alpha * 0.3) * emb + (alpha * 0.3) * match_emb
            refined = refined / (np.linalg.norm(refined) + 1e-10)

            r_t = torch.tensor(refined.astype(np.float32), device=DEVICE)
            refined_scores = self.corpus.q_embeds @ r_t
            _, refined_idx = refined_scores.topk(1)
            refined_entry = self.corpus.extract_response(
                self.corpus.entries[refined_idx[0].item()])

            if refined_entry and len(refined_entry) > 10:
                match = refined_entry
                key_words = self._extract_focus(match[:100])
                thought = f"{query} {key_words}"

            prev_emb = refined.astype(np.float32)

        final_emb = self.embeddings.embed(thought, self.tokenizer)
        if final_emb is not None:
            q_t = torch.tensor(final_emb, dtype=torch.float32, device=DEVICE)
            scores = self.corpus.q_embeds @ q_t
            _, best_idx = scores.topk(1)
            idx = best_idx[0].item()
            if idx < len(self.corpus.entries):
                result = self.corpus.extract_response(self.corpus.entries[idx])
                return result, trace, step + 1

        return None, trace, step + 1

    def _extract_focus(self, text):
        """Extract the highest-IDF words as the narrowing signal."""
        content = self.corpus.content_words(text)
        scored = []
        for w in content:
            if w in self.corpus.word_index:
                idf = np.log(self.corpus.n_entries / max(len(self.corpus.word_index[w]), 1))
                scored.append((idf, w))
        scored.sort(reverse=True)
        return ' '.join(w for _, w in scored[:5])
