"""
W*-derived Q/K projections for gradient-free attention.

The insight: Q and K should make tokens in DIFFERENT contexts produce
DIFFERENT attention patterns. We derive them from the corpus itself.

Method:
1. Find tokens that appear in multiple corpus entries
2. For each occurrence, record the token's context embedding
3. Compute W_Q, W_K such that Q·K is HIGH for same-context tokens
   and LOW for different-context tokens

This is contrastive learning via closed-form ridge regression.
No gradients. The equation:
    W* = (H'H + λI)⁻¹H'Y
"""
import os
import pickle
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_contextual_qk(embeddings, tokenizer, corpus_texts, data_dir=None,
                         n_target_tokens=1000, max_contexts=30, dim_out=32):
    """Build Q/K projections from contextual contrasts in the corpus.

    Returns W_Q, W_K each (dim, dim_out) such that:
    - tokens in similar contexts have high Q·K
    - tokens in different contexts have low Q·K

    This is what gradient descent learns in transformers.
    We compute it in closed form.
    """
    cache_path = os.path.join(data_dir, 'qk_contextual.pkl') if data_dir else None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        print(f"  Contextual Q/K: loaded from cache ({len(cached)} heads)")
        return cached

    print("  Building contextual Q/K from corpus contrasts...")
    dim = embeddings.dim

    from collections import defaultdict

    # Build 4 different Q/K pairs with different window sizes
    # Each captures a different type of contextual relationship
    window_configs = [
        (2, 'narrow'),    # Immediate neighbors — syntax
        (4, 'medium'),    # Short-range — local semantics
        (8, 'wide'),      # Paragraph-level — topic
        (16, 'global'),   # Long-range — discourse
    ]

    results = {}

    for window_size, label in window_configs:
        print(f"    Building {label} Q/K (window={window_size})...")
        token_contexts = defaultdict(list)

        n_corpus = len(corpus_texts)
        # Smaller sample for speed — 30K entries is enough with 4 scales
        sample_indices = np.random.RandomState(42 + window_size).choice(
            n_corpus, min(30000, n_corpus), replace=False)

        raw = embeddings.raw  # (vocab, dim)
        idf = embeddings.idf
        vs = embeddings.vocab_size

        for idx in sample_indices:
            text = corpus_texts[idx]
            if not isinstance(text, str) or len(text) < 20:
                continue

            ids = tokenizer.encode(text).ids
            n_ids = len(ids)
            if n_ids < 5:
                continue

            # Vectorized: get all embeddings at once
            valid_mask = np.array([0 <= t < vs for t in ids])
            if valid_mask.sum() < 3:
                continue

            ids_arr = np.array(ids)
            ids_clamped = np.clip(ids_arr, 0, vs - 1)
            embs = raw[ids_clamped]  # (n_ids, dim)
            idfs = idf[ids_clamped]  # (n_ids,)

            for pos in range(n_ids):
                if not valid_mask[pos]:
                    continue
                tid = ids[pos]
                if len(token_contexts[tid]) >= max_contexts:
                    continue

                lo = max(0, pos - window_size)
                hi = min(n_ids, pos + window_size + 1)
                # Window indices excluding pos
                w_idx = [j for j in range(lo, hi) if j != pos and valid_mask[j]]
                if len(w_idx) < 2:
                    continue

                # Distance-weighted IDF average
                w_embs = embs[w_idx]  # (w, dim)
                dists = np.abs(np.array(w_idx) - pos).astype(np.float32)
                weights = idfs[w_idx] / (1.0 + dists * 0.5)
                ctx_emb = (weights[:, None] * w_embs).sum(axis=0) / (weights.sum() + 1e-10)

                token_contexts[tid].append(ctx_emb)

        diverse_tokens = {tid: ctxs for tid, ctxs in token_contexts.items()
                          if len(ctxs) >= 5}
        print(f"      Tokens with 5+ contexts: {len(diverse_tokens)}")

        if len(diverse_tokens) < 50:
            continue

        # Build training data
        H_q, Y_q, H_k, Y_k = [], [], [], []
        selected = list(diverse_tokens.items())[:n_target_tokens]
        for tid, ctxs in selected:
            tok_emb = embeddings.raw[tid]
            for ctx_emb in ctxs:
                H_q.append(tok_emb)
                Y_q.append(ctx_emb)
                H_k.append(ctx_emb)
                Y_k.append(tok_emb)

        H_q = np.array(H_q, dtype=np.float64)
        Y_q = np.array(Y_q, dtype=np.float64)
        H_k = np.array(H_k, dtype=np.float64)
        Y_k = np.array(Y_k, dtype=np.float64)

        print(f"      Training pairs: {len(H_q)}")

        # W* = (H'H + λI)⁻¹H'Y
        lam = 1e-3  # lower regularization for tighter fit
        W_Q_full = np.linalg.solve(
            H_q.T @ H_q + lam * np.eye(dim), H_q.T @ Y_q).astype(np.float32)
        W_K_full = np.linalg.solve(
            H_k.T @ H_k + lam * np.eye(dim), H_k.T @ Y_k).astype(np.float32)

        # Reduce to dim_out via SVD
        Uq, Sq, _ = np.linalg.svd(W_Q_full, full_matrices=False)
        W_Q = (Uq[:, :dim_out] * Sq[:dim_out]).astype(np.float32)

        Uk, Sk, _ = np.linalg.svd(W_K_full, full_matrices=False)
        W_K = (Uk[:, :dim_out] * Sk[:dim_out]).astype(np.float32)

        # Measure separation
        test_tid, test_ctxs = selected[0]
        q_vec = embeddings.raw[test_tid] @ W_Q
        same = [float(np.dot(q_vec, c @ W_K)) for c in test_ctxs]
        other_tid, other_ctxs = selected[min(10, len(selected)-1)]
        diff = [float(np.dot(q_vec, c @ W_K)) for c in other_ctxs]
        sep = np.mean(same) - np.mean(diff)
        print(f"      Separation: {sep:.4f}")

        results[label] = {'W_Q': W_Q, 'W_K': W_K, 'separation': float(sep)}

    if cache_path and results:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)

    return results
