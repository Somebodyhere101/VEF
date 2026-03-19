"""
Tensor Decomposition — multi-modal word embeddings.

Instead of one embedding per word (matrix SVD), each word gets
K embeddings, one per MODE. The system prompt selects the mode.

    Matrix:  V × V  → SVD    → E[word] = 128d vector
    Tensor:  V × V × K → CP  → E[word, mode] = 128d vector per mode

"Hello" in greeting mode → conversational embedding
"Hello" in code mode → programming embedding

The mode is selected by projecting the system prompt onto the
mode vectors. No hardcoded routing — the prompt's embedding
determines which mode activates.
"""
import numpy as np
import os
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TensorEmbeddings:
    """Multi-modal embeddings via tensor decomposition."""

    def __init__(self, data_dir):
        path = os.path.join(data_dir, 'tensor_modes.npz')
        if os.path.exists(path):
            data = np.load(path)
            self.mode_embeds = data['mode_embeds']  # (K, V, d) — per-mode embeddings
            self.mode_vectors = data['mode_vectors']  # (K, d) — mode selectors
            self.K = self.mode_embeds.shape[0]
            # GPU copies
            self.mode_embeds_t = torch.tensor(
                self.mode_embeds, dtype=torch.float32, device=DEVICE)
            self.mode_vectors_t = torch.tensor(
                self.mode_vectors, dtype=torch.float32, device=DEVICE)
        else:
            self.mode_embeds = None
            self.K = 0

    @staticmethod
    def build(corpus_entries, tokenizer, base_embeddings, V, d, K=4,
              output_dir=None):
        """Build multi-modal embeddings from corpus.

        1. Classify each entry into K modes by content
        2. Build per-mode co-occurrence matrices
        3. PPMI each mode
        4. SVD each mode → per-mode embeddings
        5. Compute mode selector vectors (centroid of each mode's entries)

        Returns: TensorEmbeddings instance
        """
        import re
        import time
        from scipy import sparse
        from scipy.sparse.linalg import svds

        t0 = time.time()
        print(f"  Building tensor embeddings (V={V}, d={d}, K={K})...")

        # Step 1: Classify entries into modes
        mode_entries = [[] for _ in range(K)]  # K lists of entry indices
        CODE_WORDS = {'def', 'class', 'import', 'print', 'function', 'variable',
                      'code', 'program', 'syntax', 'compile', 'return', 'loop'}
        GREET_WORDS = {'hello', 'hi', 'hey', 'greet', 'welcome', 'morning',
                       'afternoon', 'evening', 'goodbye', 'bye', 'thanks'}

        for i, entry in enumerate(corpus_entries):
            lower = entry.lower()
            words = set(re.findall(r'[a-z]+', lower[:200]))
            code_score = len(words & CODE_WORDS)
            greet_score = len(words & GREET_WORDS)

            if code_score >= 2:
                mode_entries[1].append(i)  # mode 1: code
            elif greet_score >= 1:
                mode_entries[2].append(i)  # mode 2: greeting/conversational
            elif '?' in entry.split('Assistant:')[0] if 'Assistant:' in entry else False:
                mode_entries[3].append(i)  # mode 3: knowledge Q&A
            else:
                mode_entries[0].append(i)  # mode 0: general instruct

        for k in range(K):
            print(f"    Mode {k}: {len(mode_entries[k]):,} entries")

        # Step 2-4: Per-mode PPMI+SVD
        window = 5
        mode_embeds = np.zeros((K, V, d), dtype=np.float32)

        for k in range(K):
            entries = mode_entries[k]
            if len(entries) < 100:
                # Too few entries — use base embeddings
                mode_embeds[k] = base_embeddings.raw[:V]
                continue

            # Build co-occurrence for this mode
            cooc = np.zeros((V, V), dtype=np.float32)
            sample = entries[:min(50000, len(entries))]  # cap for speed
            batch_texts = [corpus_entries[i] for i in sample]
            encodings = tokenizer.encode_batch(batch_texts)

            for enc in encodings:
                ids = np.array(enc.ids, dtype=np.int64)
                ids = ids[(ids >= 0) & (ids < V)]
                n = len(ids)
                if n < 2:
                    continue
                for offset in range(1, window + 1):
                    if offset >= n:
                        break
                    rows = ids[:n-offset]
                    cols = ids[offset:]
                    np.add.at(cooc, (rows, cols), 1.0)
                    np.add.at(cooc, (cols, rows), 1.0)

            # PPMI (vectorized)
            M = sparse.csr_matrix(cooc)
            del cooc
            total = M.sum()
            if total < 100:
                mode_embeds[k] = base_embeddings.raw[:V]
                continue

            row_sum = np.array(M.sum(axis=1)).flatten()
            col_sum = np.array(M.sum(axis=0)).flatten()
            M_coo = M.tocoo()
            pmi = M_coo.data.copy().astype(np.float64)
            rs = np.maximum(row_sum[M_coo.row], 1e-10)
            cs = np.maximum(col_sum[M_coo.col], 1e-10)
            pmi = np.maximum(np.log(pmi * total / (rs * cs)), 0.0)
            ppmi = sparse.csr_matrix((pmi, (M_coo.row, M_coo.col)), shape=(V, V))

            # SVD
            k_svd = min(d, ppmi.shape[0] - 1, ppmi.shape[1] - 1)
            if k_svd < d:
                mode_embeds[k] = base_embeddings.raw[:V]
                continue

            try:
                U, S, Vt = svds(ppmi.astype(np.float64), k=k_svd)
                order = np.argsort(-S)
                U = U[:, order]; S = S[order]
                emb = (U * np.sqrt(S)).astype(np.float32)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1
                mode_embeds[k] = emb / norms
            except:
                mode_embeds[k] = base_embeddings.raw[:V]

            print(f"    Mode {k}: SVD done")

        # Step 5: Mode selector vectors
        # Each mode's vector = centroid of its entries' embeddings
        mode_vectors = np.zeros((K, d), dtype=np.float32)
        idf = base_embeddings.idf
        raw = base_embeddings.raw

        for k in range(K):
            entries = mode_entries[k][:10000]
            if not entries:
                continue
            batch = [corpus_entries[i][:200] for i in entries]
            encodings = tokenizer.encode_batch(batch)
            acc = np.zeros(d, dtype=np.float64)
            count = 0
            for enc in encodings:
                ids = [t for t in enc.ids if 0 <= t < V]
                if not ids:
                    continue
                w = np.array([idf[t] for t in ids])
                total_w = w.sum()
                if total_w < 1e-10:
                    continue
                emb = (raw[ids] * w[:, None]).sum(0) / total_w
                acc += emb
                count += 1
            if count > 0:
                mode_vectors[k] = (acc / count).astype(np.float32)
                norm = np.linalg.norm(mode_vectors[k])
                if norm > 1e-10:
                    mode_vectors[k] /= norm

        # Save
        if output_dir:
            np.savez(os.path.join(output_dir, 'tensor_modes.npz'),
                     mode_embeds=mode_embeds, mode_vectors=mode_vectors)

        print(f"  Tensor embeddings built ({time.time()-t0:.0f}s)")

        te = TensorEmbeddings.__new__(TensorEmbeddings)
        te.mode_embeds = mode_embeds
        te.mode_vectors = mode_vectors
        te.K = K
        te.mode_embeds_t = torch.tensor(mode_embeds, dtype=torch.float32, device=DEVICE)
        te.mode_vectors_t = torch.tensor(mode_vectors, dtype=torch.float32, device=DEVICE)
        return te

    def embed(self, text, tokenizer, idf, mode_emb=None):
        """Embed text using mode-weighted combination.

        If mode_emb is provided (e.g., system prompt embedding),
        it selects which mode(s) to use via dot product similarity.
        """
        if self.mode_embeds is None or self.K == 0:
            return None

        ids = [t for t in tokenizer.encode(text).ids
               if 0 <= t < self.mode_embeds.shape[1]]
        if not ids:
            return None

        # Compute mode weights from the mode selector
        if mode_emb is not None:
            mode_t = torch.tensor(mode_emb, dtype=torch.float32, device=DEVICE)
            weights = torch.softmax(self.mode_vectors_t @ mode_t * 5.0, dim=0)
        else:
            weights = torch.ones(self.K, device=DEVICE) / self.K

        # Weighted combination of per-mode embeddings for each token
        ids_t = torch.tensor(ids, dtype=torch.long, device=DEVICE)
        idf_w = torch.tensor([idf[t] for t in ids], dtype=torch.float32, device=DEVICE)
        total_idf = idf_w.sum()
        if total_idf < 1e-10:
            return None

        # (K, len_ids, d) — each mode's embedding for each token
        token_embs = self.mode_embeds_t[:, ids_t, :]  # (K, n_tokens, d)

        # Weight by mode: (K, n_tokens, d) × (K, 1, 1) → sum over K → (n_tokens, d)
        weighted = (token_embs * weights.view(self.K, 1, 1)).sum(0)  # (n_tokens, d)

        # IDF pool
        emb = (weighted * idf_w.unsqueeze(1)).sum(0) / total_idf
        emb = emb / (emb.norm() + 1e-10)
        return emb.cpu().numpy().astype(np.float32)
