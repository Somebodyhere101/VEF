"""
Introspection — the model measures its own confidence.

When retrieval returns results, the model checks: does this actually
answer what was asked? It measures query-response alignment and
compares to its corpus baseline. If the match isn't significantly
above random, the model knows it doesn't know.

Also handles typo correction: when confidence is low, the model
searches for visually similar known words (edit distance).
"""
import re
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Introspection:

    def __init__(self, embeddings, corpus, tokenizer):
        self.embeddings = embeddings
        self.corpus = corpus
        self.tokenizer = tokenizer

    def measure_confidence(self, query, best_response, best_score):
        """How confident is the model that this response answers the query?

        Returns (confidence, details) where confidence is a float.
        The model computes this from its OWN embeddings — no threshold needed.
        High = confident. Low = uncertain.
        """
        q_emb = self.embeddings.embed(query, self.tokenizer)
        r_emb = self.embeddings.embed(best_response[:200], self.tokenizer)

        if q_emb is None or r_emb is None:
            return 0.0, "Could not embed query or response"

        alignment = float(np.dot(q_emb, r_emb))

        # Compare to corpus baseline
        q_t = torch.tensor(q_emb, dtype=torch.float32, device=DEVICE)
        all_scores = self.corpus.q_embeds @ q_t
        baseline = float(all_scores.mean())
        margin = best_score - baseline

        confidence = alignment * margin
        details = (f"alignment={alignment:.3f}, margin={margin:.3f} "
                   f"(best={best_score:.3f}, baseline={baseline:.3f})")

        return confidence, details

    def try_spell_correction(self, query, concept_words):
        """If unknown words look like known words, correct and retry.

        Like a human sounding out "hazmlloe" → "hello". Uses edit
        distance (visual similarity), not embedding similarity.
        """
        corrections = {}
        for word in concept_words:
            # Skip words that are well-established in the corpus (>50 entries)
            if word in self.corpus.word_index and len(self.corpus.word_index[word]) > 50:
                continue

            candidates = []
            word_chars = set(word)

            for known in self.corpus.word_index:
                if abs(len(known) - len(word)) > 2:
                    continue
                if len(word_chars & set(known)) < len(word) // 2:
                    continue
                dist = self._edit_distance(word, known)
                if dist <= 2 and dist > 0:
                    freq = len(self.corpus.word_index[known])
                    candidates.append((dist, freq, known))

            if candidates:
                # Score: distance penalized, frequency rewarded.
                # A very common word at dist=2 beats a rare word at dist=1.
                # score = -dist + log2(freq)/10
                import math
                scored = []
                for dist, freq, w in candidates:
                    s = -dist + math.log2(max(freq, 1)) / 5.0
                    scored.append((s, w, dist))
                scored.sort(reverse=True)
                best_match = scored[0][1]
                best_dist = scored[0][2]
                corrections[word] = (best_match, best_dist)

        return corrections

    def find_partial_knowledge(self, concept_words, retrieval):
        """What does the model know about each concept individually?"""
        known = []
        for word in concept_words:
            results = retrieval.search(word, top_k=1)
            if results:
                resp = results[0][1]
                if word in resp.lower()[:150]:
                    known.append((word, resp[:150]))
        return known

    @staticmethod
    def _edit_distance(a, b):
        n, m = len(a), len(b)
        if n == 0: return m
        if m == 0: return n
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            curr = [i] + [0] * m
            for j in range(1, m + 1):
                cost = 0 if a[i-1] == b[j-1] else 1
                curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
            prev = curr
        return prev[m]
