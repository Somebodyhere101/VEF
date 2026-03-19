"""
Retrieval — Adaptive Instruct Scoring
=======================================
Three signals, adaptively weighted by query confidence:
  1. Q-Q cosine: question-question similarity (IDF-weighted)
  2. Response-blended: response match to system-blended query
  3. Response-system: response style alignment with instruction

High confidence (specific query) → Q-Q dominates
Low confidence (ambiguous/short query) → response-blended dominates

Plus:
  - Lateral inhibition (suppress noise, amplify standouts)
  - Quality filtering (prefer well-formed responses)
  - System prompt injection (model can describe itself)
"""
import re
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Retrieval:

    def __init__(self, embeddings, corpus, tokenizer, attention=None):
        self.embeddings = embeddings
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.attention = attention
        self._system_emb = None
        self._system_entries = []
        self._resp_embeds = None
        self._conv_mask = None

    def _load_conv_mask(self):
        """Load conversation mask for quality filtering."""
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'conv_mask.npy')
        if os.path.exists(path):
            self._conv_mask = np.load(path)
            self._conv_indices = torch.tensor(
                np.where(self._conv_mask)[0], dtype=torch.long, device=DEVICE)

    def _load_resp_embeds(self):
        """Lazy-load response embeddings."""
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'resp_embeds_idf.npy')
        if os.path.exists(path):
            r = np.load(path)
            self._resp_embeds = torch.tensor(r.astype(np.float32), device=DEVICE)

    def set_system(self, prompt):
        """Set system prompt for adaptive scoring + injection."""
        self._system_emb = None
        self._system_entries = []
        if prompt:
            self._system_emb = self.embeddings.embed(prompt, self.tokenizer)
            # Identity triggers: ONLY the question part, matched against queries.
            # The prompt is the response. Tight matching prevents over-triggering.
            identity_questions = [
                "Who are you",
                "What is your name",
                "What are you",
                "Tell me about yourself",
                "Describe yourself",
            ]
            for q in identity_questions:
                emb = self.embeddings.embed(q, self.tokenizer)
                if emb is not None:
                    self._system_entries.append((emb, prompt))

    def search(self, query, top_k=30):
        """Adaptive instruct scoring with three signals."""
        # Static embedding for primary retrieval (matches corpus embedding space)
        q_emb = self.embeddings.embed(query, self.tokenizer)
        if q_emb is None:
            return []

        # Contextual embedding for re-ranking (attention captures word order)
        q_emb_ctx = self.embeddings.embed(query, self.tokenizer, attention=self.attention)

        # Lazy-load response embeddings and conversation mask
        if self._resp_embeds is None:
            self._load_resp_embeds()
        if self._conv_mask is None:
            self._load_conv_mask()

        q_t = torch.tensor(q_emb, dtype=torch.float32, device=DEVICE)
        content_words = self.corpus.content_words(query)
        n_words = len(query.split())

        # For short queries (1-3 words): search ONLY conversational entries.
        # This prevents code/wiki fragments from drowning out natural responses.
        # Like the brain filtering modality — "Hello" activates social circuits, not code.
        use_conv_only = (n_words <= 3 and self._conv_mask is not None
                         and not any(c.isdigit() for c in query))

        # === Signal 1: Q-Q cosine (question-question similarity) ===
        if content_words:
            candidates = set()
            for w in content_words:
                candidates.update(
                    ci for ci in self.corpus.word_index[w]
                    if ci < self.corpus.n_entries)

            # For short queries, filter to conversational entries
            if use_conv_only:
                candidates = {ci for ci in candidates
                              if ci < len(self._conv_mask) and self._conv_mask[ci]}

            if len(candidates) > 50:
                cand_idx = torch.tensor(sorted(candidates), dtype=torch.long, device=DEVICE)
                qq_scores = self.corpus.q_embeds[cand_idx] @ q_t
                k = min(top_k * 4, len(cand_idx))
                top_qq, top_local = qq_scores.topk(k)
                top_idx = cand_idx[top_local]
            else:
                # No content words or too few candidates — full search
                if use_conv_only and hasattr(self, '_conv_indices'):
                    qq_scores = self.corpus.q_embeds[self._conv_indices] @ q_t
                    k = min(top_k * 4, len(self._conv_indices))
                    top_qq, top_local = qq_scores.topk(k)
                    top_idx = self._conv_indices[top_local]
                else:
                    qq_scores = self.corpus.q_embeds @ q_t
                    top_qq, top_idx = qq_scores.topk(min(top_k * 4, len(qq_scores)))
        else:
            if use_conv_only and hasattr(self, '_conv_indices'):
                qq_scores = self.corpus.q_embeds[self._conv_indices] @ q_t
                k = min(top_k * 4, len(self._conv_indices))
                top_qq, top_local = qq_scores.topk(k)
                top_idx = self._conv_indices[top_local]
            else:
                qq_scores = self.corpus.q_embeds @ q_t
                top_qq, top_idx = qq_scores.topk(min(top_k * 4, len(qq_scores)))

        # === Signal 2: Response-blended scoring ===
        # Blend query with system prompt, match against RESPONSE embeddings
        rb_scores = torch.zeros_like(top_qq)
        if self._resp_embeds is not None and self._resp_embeds.shape[0] == self.corpus.n_entries:
            blend_weight = min(0.5, 0.7 / max(n_words, 1))
            if self._system_emb is not None:
                blended = q_emb * (1 - blend_weight) + self._system_emb * blend_weight
                blended = blended / (np.linalg.norm(blended) + 1e-10)
            else:
                blended = q_emb
            bl_t = torch.tensor(blended, dtype=torch.float32, device=DEVICE)
            rb_scores = self._resp_embeds[top_idx] @ bl_t

        # === Signal 3: Response-system alignment ===
        rs_scores = torch.zeros_like(top_qq)
        if self._system_emb is not None and self._resp_embeds is not None:
            sys_t = torch.tensor(self._system_emb, dtype=torch.float32, device=DEVICE)
            rs_scores = self._resp_embeds[top_idx] @ sys_t

        # === Adaptive weighting ===
        max_qq = float(top_qq.max()) if len(top_qq) > 0 else 0
        confidence = max(0, min(1, (max_qq - 0.6) / 0.3))
        confidence *= min(1.0, n_words / 4.0)

        w_qq = 0.10 + 0.70 * confidence
        w_rb = 0.85 - 0.60 * confidence
        w_rs = 0.05 - 0.10 * confidence
        w_rs = max(0.0, w_rs)

        # Normalize signals to same scale before combining.
        # Without this, Q-Q (~0.9 range) dominates RB (~0.5 range)
        # regardless of weights. Normalization lets the weights actually matter.
        def _norm(t):
            mn, mx = t.min(), t.max()
            return (t - mn) / (mx - mn + 1e-10) if mx > mn else t

        final = w_qq * _norm(top_qq) + w_rb * _norm(rb_scores) + w_rs * _norm(rs_scores)

        # === Build result list ===
        results = []

        # System prompt injection — only when query is genuinely about identity.
        # Must score ABOVE 0.85 similarity to a trigger question AND
        # the corpus's best result must not already be a good answer.
        if self._system_entries:
            best_sim = max(float(np.dot(q_emb, emb)) for emb, _ in self._system_entries)
            corpus_best = float(top_qq[0]) if len(top_qq) > 0 else 0
            if best_sim > 0.9 or (best_sim > 0.8 and corpus_best < best_sim * 0.5):
                results.append((best_sim * 1.5, self._system_entries[0][1]))

        for score, idx in zip(final.tolist(), top_idx.tolist()):
            if idx >= len(self.corpus.entries):
                continue
            resp = self.corpus.extract_response(self.corpus.entries[idx])
            if not resp or len(resp) < 10:
                continue

            # Content word promotion
            if content_words:
                resp_lower = resp.lower()
                n_found = sum(1 for w in content_words if w in resp_lower)
                score = score + score * 0.3 * (n_found / len(content_words))

            # Quality: detect and suppress code, URLs, markup
            resp_lower = resp.lower() if not content_words else resp_lower
            is_code = (
                resp.startswith(('http', '{', '<', 'def ', 'class ', 'import ',
                                 '#!', '#', '//', '/*', '```')) or
                'console.log' in resp or 'println' in resp or
                'function(' in resp or '```' in resp or
                'System.out' in resp or 'print(' in resp or
                'named "say_' in resp or 'named "get_' in resp or
                ('the function' in resp_lower and 'print' in resp_lower) or
                resp.count('{') > 2 or resp.count('(') > 5 or
                resp.count('=') > 3
            )
            # For short queries: suppress responses ABOUT code/programming
            # even if they're natural language descriptions.
            # "There are no errors in the code" passes code detection but
            # is wrong for conversational queries like "Hello"
            is_code_discussion = (n_words <= 3 and (
                sum(1 for w in ['code', 'function', 'output', 'print', 'error',
                    'variable', 'string', 'program', 'bug', 'syntax', 'compile',
                    'return', 'method', 'class', 'loop', 'array', 'snippet']
                    if w in resp_lower) >= 2))
            quality = min(len(resp) / 200.0, 1.0)
            if is_code:
                quality *= 0.15
            elif is_code_discussion:
                quality *= 0.2
            score = score * (0.6 + 0.4 * quality)

            # Length matching: short queries → prefer concise responses
            if n_words <= 3 and len(resp) > 400:
                score *= 0.6

            results.append((score, resp))

        # === Lateral inhibition ===
        if len(results) > 3:
            scores_only = [s for s, _ in results]
            mean_s = np.mean(scores_only)
            std_s = np.std(scores_only) + 1e-10
            inhibited = []
            for score, resp in results:
                z = (score - mean_s) / std_s
                if z > 0:
                    adj = score * (1.0 + 0.15 * z)
                else:
                    adj = score * max(0.4, 1.0 + 0.2 * z)
                inhibited.append((adj, resp))
            results = inhibited

        results.sort(key=lambda x: -x[0])
        return results[:top_k]
