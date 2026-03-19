"""
VEF Train — Build the model from raw text data.
================================================
Turns text files into a complete VEF model. No gradient descent.

Usage:
    python train.py --input C:/path/to/texts/       # directory of .txt files
    python train.py --input data.txt                 # single file
    python train.py --input C:/path/ --append        # add to existing model

Pipeline (all closed-form, GPU-accelerated):
    1. Tokenize    — BPE tokenizer (Rust-parallel)
    2. Co-occur    — GPU scatter_add (window=5)
    3. PPMI        — Vectorized (no Python loops)
    4. SVD         — torch.svd_lowrank (GPU) or scipy (CPU)
    5. IDF         — Computed from cached token IDs (no re-tokenization)
    6. Corpus      — Parse Q/A pairs
    7. Embed       — Batch GPU matrix multiply (not per-entry)
    8. Index       — Inverted word index + conversation mask
    9. Mine        — Arithmetic, definitions, categories
"""
import argparse
import glob
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing.pool import ThreadPool

import numpy as np

try:
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HAS_GPU = DEVICE.type == 'cuda'
except ImportError:
    DEVICE = None
    HAS_GPU = False

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as BLPre
from tokenizers.decoders import ByteLevel as BLDec

# Pre-compiled patterns
WORD_RE = re.compile(r'[a-z]+')
DEF_RE = re.compile(r'human:\s*what(?:\'s| is| are) (?:a |an |the )?(.+?)[\?\n]')
ARITH_PATTERNS = {
    '+': re.compile(r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)'),
    '-': re.compile(r'(\d+)\s*-\s*(\d+)\s*=\s*(\d+)'),
    '*': re.compile(r'(\d+)\s*[*×x]\s*(\d+)\s*=\s*(\d+)'),
}


STEP = [0]

def step(name):
    """Print a clear step header."""
    STEP[0] += 1
    elapsed = time.time() - T0[0] if T0[0] else 0
    print(f"\n  ┌─ Step {STEP[0]}: {name} ({elapsed:.0f}s elapsed)")

def log(msg):
    print(f"  │ {msg}")

def done(msg=""):
    print(f"  └─ {msg}")

def progress(current, total, label="", interval=5):
    """Print progress at ~interval% increments. No flooding."""
    pct = 100 * current / max(total, 1)
    step_pct = max(interval, 1)
    if current == total or (int(pct) % step_pct == 0 and int(100 * (current - 1) / max(total, 1)) % step_pct != 0):
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"  │ {bar} {pct:5.1f}% ({current:,}/{total:,}) {label}", flush=True)

T0 = [None]  # mutable for closure


# ═══════════════════════════════════════════════════════════
# STEP 1: LOAD DATA (parallel file I/O)
# ═══════════════════════════════════════════════════════════

def _load_one_file(fname):
    """Load and parse a single text file."""
    with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    entries = []
    for chunk in re.split(r'<\|endoftext\|>|\n\n\n+', content):
        chunk = chunk.strip()
        if len(chunk) > 50:
            chunk = re.sub(r'###\s*Instruction:\s*', 'Human: ', chunk)
            chunk = re.sub(r'###\s*Response:\s*', '\nAssistant: ', chunk)
            entries.append(chunk)
    return entries


def load_texts(path):
    """Load texts from file or directory — parallel I/O."""
    texts = []
    if os.path.isdir(path):
        fnames = sorted(glob.glob(os.path.join(path, '*.txt')))
        step(f"Load {len(fnames)} text files (parallel I/O)")
        with ThreadPool(8) as pool:
            for i, entries in enumerate(pool.imap(_load_one_file, fnames)):
                texts.extend(entries)
                progress(i + 1, len(fnames), f"{len(texts):,} entries")
        done(f"{len(texts):,} texts loaded")
    elif os.path.isfile(path):
        step("Load text file")
        texts = _load_one_file(path)
        done(f"{len(texts):,} texts loaded")
    return texts


# ═══════════════════════════════════════════════════════════
# STEP 2: TOKENIZER
# ═══════════════════════════════════════════════════════════

def build_tokenizer(texts, V, output_dir):
    """Train BPE tokenizer."""
    path = os.path.join(output_dir, 'tokenizer.json')
    if os.path.exists(path):
        log("Tokenizer exists, loading...")
        tok = Tokenizer.from_file(path)
        tok.decoder = BLDec()
        if tok.get_vocab_size() != V:
            log(f"Vocab size mismatch: cached={tok.get_vocab_size()}, requested={V}. Rebuilding...")
        else:
            return tok
    t0 = time.time()
    step(f"Train BPE tokenizer (V={V})")
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = BLPre()
    tok.decoder = BLDec()
    trainer = BpeTrainer(vocab_size=V, min_frequency=2,
                         special_tokens=["<|endoftext|>"])
    tok.train_from_iterator(texts[:min(100000, len(texts))], trainer=trainer)
    tok.save(path)
    done(f"{tok.get_vocab_size()} tokens ({time.time()-t0:.0f}s)")
    return tok


# ═══════════════════════════════════════════════════════════
# STEP 3-5: CO-OCCUR → PPMI → SVD → IDF (single pass)
# ═══════════════════════════════════════════════════════════

def build_embeddings(texts, tok, V, d, output_dir):
    """PPMI+SVD embeddings — all optimizations applied.

    Optimizations vs naive:
      - Batch tokenization (encode_batch, 50K per batch)
      - GPU scatter_add for co-occurrence
      - Vectorized PPMI (no Python loop)
      - torch.svd_lowrank on GPU (vs scipy on CPU)
      - IDF from cached token sets (no re-tokenization)
    """
    emb_path = os.path.join(output_dir, 'token_embeds.npy')
    idf_path = os.path.join(output_dir, 'idf_weights.npy')
    if os.path.exists(emb_path) and os.path.exists(idf_path):
        log("Embeddings exist, loading...")
        cached_emb = np.load(emb_path)
        cached_idf = np.load(idf_path)
        if cached_emb.shape[1] != d:
            log(f"Embedding dim mismatch: cached={cached_emb.shape[1]}, requested={d}. Rebuilding...")
            del cached_emb, cached_idf
        else:
            return cached_emb, cached_idf

    from scipy import sparse

    window = 5
    t0 = time.time()

    # === CO-OCCURRENCE (GPU scatter_add, large batches) ===
    step(f"Co-occurrence matrix (V={V}, window={window}, GPU={HAS_GPU})")
    use_gpu_dense = HAS_GPU and V <= 30000
    if use_gpu_dense:
        cooc_gpu = torch.zeros(V, V, dtype=torch.float32, device=DEVICE)
        cooc_flat = cooc_gpu.view(-1)
    else:
        cooc = defaultdict(float)

    n_tokens = 0
    doc_freq = np.zeros(V, dtype=np.float64)  # IDF computed HERE, not second pass
    batch_size = 50000  # large batches = fewer GPU transfers

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]
        encodings = tok.encode_batch(batch)

        # Per-text: compute flat co-occurrence indices in numpy (fast, vectorized)
        # Accumulate ALL pairs from the batch, then ONE GPU scatter
        all_flat_fwd, all_flat_bwd, all_weights = [], [], []
        for enc in encodings:
            ids = np.array(enc.ids, dtype=np.int64)
            ids = ids[(ids >= 0) & (ids < V)]
            n = len(ids)
            if n < 2:
                continue
            n_tokens += n
            doc_freq[np.unique(ids)] += 1

            # Vectorized pair computation per text (5 numpy operations, not loops)
            for offset in range(1, window + 1):
                if offset >= n:
                    break
                weight = 1.0 / offset
                if use_gpu_dense:
                    all_flat_fwd.append(ids[:n-offset] * V + ids[offset:])
                    all_flat_bwd.append(ids[offset:] * V + ids[:n-offset])
                    n_pairs = n - offset
                    all_weights.append(np.full(n_pairs, weight, dtype=np.float32))
                else:
                    for r, c in zip(ids[:n-offset], ids[offset:]):
                        cooc[(int(r), int(c))] += weight
                        cooc[(int(c), int(r))] += weight

        # ONE GPU scatter per batch (not per text)
        if use_gpu_dense and all_flat_fwd:
            flat_fwd = torch.tensor(np.concatenate(all_flat_fwd), dtype=torch.long, device=DEVICE)
            flat_bwd = torch.tensor(np.concatenate(all_flat_bwd), dtype=torch.long, device=DEVICE)
            weights_t = torch.tensor(np.concatenate(all_weights), dtype=torch.float32, device=DEVICE)
            cooc_flat.scatter_add_(0, flat_fwd, weights_t)
            cooc_flat.scatter_add_(0, flat_bwd, weights_t)
            del flat_fwd, flat_bwd, weights_t

        progress(min(batch_start + batch_size, len(texts)), len(texts),
                 f"{n_tokens/1e6:.0f}M tokens", interval=10)

    done(f"{n_tokens/1e6:.0f}M tokens ({time.time()-t0:.0f}s)")

    # === PPMI (vectorized — no Python loop) ===
    t1 = time.time()
    step("PPMI (vectorized, no Python loops)")
    if use_gpu_dense:
        cooc_cpu = cooc_gpu.cpu().numpy()
        rows, cols = np.nonzero(cooc_cpu)
        data = cooc_cpu[rows, cols].astype(np.float64)
        M = sparse.csr_matrix((data, (rows, cols)), shape=(V, V))
        del cooc_gpu, cooc_cpu
        torch.cuda.empty_cache()
    else:
        r, c, d_list = [], [], []
        for (i, j), v in cooc.items():
            r.append(i); c.append(j); d_list.append(v)
        M = sparse.csr_matrix((d_list, (r, c)), shape=(V, V))

    total = M.sum()
    row_sum = np.array(M.sum(axis=1)).flatten()
    col_sum = np.array(M.sum(axis=0)).flatten()
    M_coo = M.tocoo()

    # VECTORIZED PPMI: replaces the Python for-loop
    pmi_data = M_coo.data.copy().astype(np.float64)
    row_s = np.maximum(row_sum[M_coo.row], 1e-10)
    col_s = np.maximum(col_sum[M_coo.col], 1e-10)
    pmi_data = np.log(pmi_data * total / (row_s * col_s))
    pmi_data = np.maximum(pmi_data, 0.0)

    ppmi = sparse.csr_matrix((pmi_data, (M_coo.row, M_coo.col)), shape=(V, V))
    del M, M_coo
    done(f"{(pmi_data > 0).sum():,} positive entries ({time.time()-t1:.0f}s)")

    # === SVD (GPU if available) ===
    t2 = time.time()
    step(f"SVD → {d}d embeddings")
    if HAS_GPU:
        try:
            ppmi_dense = torch.tensor(ppmi.toarray(), dtype=torch.float32, device=DEVICE)
            U_t, S_t, V_t = torch.svd_lowrank(ppmi_dense, q=d)
            U = U_t.cpu().numpy()
            S = S_t.cpu().numpy()
            del ppmi_dense, U_t, S_t, V_t
            torch.cuda.empty_cache()
            done(f"SVD (GPU) ({time.time()-t2:.0f}s)")
        except RuntimeError:
            log("  GPU SVD failed (OOM?), falling back to CPU...")
            from scipy.sparse.linalg import svds
            U, S, Vt = svds(ppmi.astype(np.float64), k=d)
            order = np.argsort(-S); U = U[:, order]; S = S[order]
            done(f"SVD (CPU) ({time.time()-t2:.0f}s)")
    else:
        from scipy.sparse.linalg import svds
        U, S, Vt = svds(ppmi.astype(np.float64), k=d)
        order = np.argsort(-S); U = U[:, order]; S = S[order]
        log(f"  SVD (CPU) done ({time.time()-t2:.0f}s)")

    del ppmi
    embeds = (U * np.sqrt(S)).astype(np.float32)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeds = embeds / norms
    np.save(emb_path, embeds)

    # === IDF (from cached doc_freq — no re-tokenization) ===
    idf = np.log(len(texts) / np.maximum(doc_freq, 1)).astype(np.float32)
    np.save(idf_path, idf)

    log(f"Embeddings: {embeds.shape}, total: {time.time()-t0:.0f}s")
    return embeds, idf


# ═══════════════════════════════════════════════════════════
# BATCH EMBEDDING (GPU matrix multiply, not per-entry)
# ═══════════════════════════════════════════════════════════

def batch_embed(texts, tok, E, idf_weights, V, d):
    """Embed all texts via batch tokenization + GPU matrix multiply.

    Instead of 1.42M individual embed() calls:
      1. encode_batch() → token ID matrices
      2. GPU gather: E[token_ids] → (batch, max_len, dim)
      3. IDF-weighted sum on GPU
      4. Normalize

    ~10x faster than per-entry embedding.
    """
    N = len(texts)
    result = np.zeros((N, d), dtype=np.float32)
    E_t = torch.tensor(E, dtype=torch.float32, device=DEVICE) if HAS_GPU else None
    idf_t = torch.tensor(idf_weights, dtype=torch.float32, device=DEVICE) if HAS_GPU else None
    batch_size = 4096

    for batch_start in range(0, N, batch_size):
        batch = texts[batch_start:batch_start + batch_size]
        encodings = tok.encode_batch(batch)

        # Extract token IDs, find max length
        all_ids = []
        max_len = 0
        for enc in encodings:
            ids = [t for t in enc.ids if 0 <= t < V]
            all_ids.append(ids)
            max_len = max(max_len, len(ids))

        if max_len == 0:
            continue

        bs = len(batch)
        # Pad to max_len
        token_matrix = np.zeros((bs, max_len), dtype=np.int64)
        mask = np.zeros((bs, max_len), dtype=np.float32)
        for i, ids in enumerate(all_ids):
            if ids:
                token_matrix[i, :len(ids)] = ids
                mask[i, :len(ids)] = 1.0

        if HAS_GPU:
            tok_t = torch.tensor(token_matrix, dtype=torch.long, device=DEVICE)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE)

            # Gather embeddings: (bs, max_len, d)
            tok_embs = E_t[tok_t]
            # IDF weights: (bs, max_len)
            weights = idf_t[tok_t] * mask_t
            # Weighted sum: (bs, d)
            total_w = weights.sum(dim=1, keepdim=True).clamp(min=1e-10)
            emb = (tok_embs * weights.unsqueeze(2)).sum(dim=1) / total_w
            # Normalize
            emb = emb / (emb.norm(dim=1, keepdim=True).clamp(min=1e-10))
            result[batch_start:batch_start + bs] = emb.cpu().numpy()
        else:
            # CPU fallback
            for i, ids in enumerate(all_ids):
                if not ids:
                    continue
                w = np.array([idf_weights[t] for t in ids])
                total = w.sum()
                if total < 1e-10:
                    continue
                emb = (E[ids] * w[:, None]).sum(0) / total
                norm = np.linalg.norm(emb)
                if norm > 1e-10:
                    result[batch_start + i] = emb / norm

        progress(min(batch_start + batch_size, N), N, "", interval=10)

    return result


# ═══════════════════════════════════════════════════════════
# STEP 6-9: CORPUS + EMBED + INDEX + MINE
# ═══════════════════════════════════════════════════════════

def build_corpus_and_index(texts, tok, embeds, idf, V, d, output_dir):
    """Build all derived data from corpus."""
    t0 = time.time()

    # Save corpus
    log(f"Saving corpus ({len(texts):,} entries)...")
    with open(os.path.join(output_dir, 'corpus_texts.pkl'), 'wb') as f:
        pickle.dump(texts, f)

    if tok.decoder is None:
        tok.decoder = BLDec()

    # Query embeddings (batch GPU)
    t1 = time.time()
    step(f"Embed {len(texts):,} corpus entries (batch GPU)")
    q_embeds = batch_embed(texts, tok, embeds, idf, V, d)
    np.save(os.path.join(output_dir, 'q_embeds_idf.npy'), q_embeds)
    done(f"Query embeds ({time.time()-t1:.0f}s)")

    # Response embeddings (batch GPU)
    t2 = time.time()
    step("Embed responses (batch GPU)")
    responses = []
    for text in texts:
        if 'Assistant:' in text:
            responses.append(text.split('Assistant:', 1)[1].strip()[:200])
        else:
            lines = text.strip().split('\n')
            responses.append('\n'.join(lines[1:]).strip()[:200] if len(lines) >= 2 else text[:200])
    r_embeds = batch_embed(responses, tok, embeds, idf, V, d)
    np.save(os.path.join(output_dir, 'resp_embeds_idf.npy'), r_embeds)
    del responses
    done(f"Response embeds ({time.time()-t2:.0f}s)")

    # Word index
    t3 = time.time()
    step("Build word index")
    word_index = defaultdict(list)
    for i, text in enumerate(texts):
        words = set(WORD_RE.findall(text.lower()))
        for w in words:
            if len(w) >= 2:
                word_index[w].append(i)
    with open(os.path.join(output_dir, 'word_index.pkl'), 'wb') as f:
        pickle.dump(dict(word_index), f)
    done(f"{len(word_index):,} words ({time.time()-t3:.0f}s)")

    # Conversation mask + definitions + arithmetic in one pass
    t4 = time.time()
    step("Mine: conversation mask + definitions + arithmetic")
    code_starts = ('def ', 'class ', 'import ', '#', '//', '/*', '{', '<', '```', 'http')
    conv_mask = np.zeros(len(texts), dtype=bool)
    for i, e in enumerate(texts):
        if 'Assistant:' not in e:
            continue
        resp = e.split('Assistant:', 1)[1].strip()
        if len(resp) < 15:
            continue
        if any(resp.startswith(p) for p in code_starts):
            continue
        if 'print(' in resp[:100] or 'console.log' in resp[:100] or '```' in resp[:100]:
            continue
        if resp.count('{') > 3 or resp.count('(') > 8:
            continue
        if resp[0].isalpha() or resp[0].isdigit() or resp[0] in '"\'':
            conv_mask[i] = True
    np.save(os.path.join(output_dir, 'conv_mask.npy'), conv_mask)
    log(f"Conv mask: {conv_mask.sum():,}/{len(texts):,}")

    # Definitions (pre-filtered)
    definitions = {}
    for text in texts:
        if 'Assistant:' not in text or 'what' not in text.lower():
            continue
        m = DEF_RE.match(text.lower())
        if m:
            subject = m.group(1).strip()
            resp = text.split('Assistant:', 1)[1].strip()
            first = re.split(r'[.!?\n]', resp)[0].strip()
            if len(first) > 20 and subject not in definitions:
                definitions[subject] = first + '.'
    with open(os.path.join(output_dir, 'definitions.pkl'), 'wb') as f:
        pickle.dump(definitions, f)
    log(f"Definitions: {len(definitions):,}")

    # Arithmetic facts (pre-filtered)
    facts = {}
    for text in texts:
        if '=' not in text:
            continue
        for op_sym, pat in ARITH_PATTERNS.items():
            for m in pat.finditer(text):
                a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if a > 100 or b > 100 or c > 10000:
                    continue
                key = (a, op_sym, b)
                if key not in facts:
                    facts[key] = Counter()
                facts[key][c] += 1
    with open(os.path.join(output_dir, 'arithmetic_facts.pkl'), 'wb') as f:
        pickle.dump(facts, f)
    log(f"Arithmetic: {len(facts):,} facts")
    done(f"Mining complete ({time.time()-t4:.0f}s)")
    return len(texts)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VEF Train — Build a model from text data",
        epilog="Example: python train.py --input C:/data/instruct/")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/')
    parser.add_argument('--vocab', type=int, default=50000)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    t_total = time.time()
    T0[0] = t_total

    print(f"\n{'='*50}")
    print(f"  VEF Train — Zero Gradient Descent")
    print(f"{'='*50}")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Vocab:  {args.vocab}, Dim: {args.dim}")
    if HAS_GPU:
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print()

    texts = load_texts(args.input)
    if not texts:
        print("ERROR: No texts found."); sys.exit(1)

    if args.append:
        existing_path = os.path.join(args.output, 'corpus_texts.pkl')
        if os.path.exists(existing_path):
            with open(existing_path, 'rb') as f:
                existing = pickle.load(f)
            log(f"Appending to {len(existing):,} existing entries")
            existing_set = set(e[:80] for e in existing)
            new = [t for t in texts if t[:80] not in existing_set]
            texts = existing + new
            log(f"Merged: {len(texts):,} total")
            for fn in ['q_embeds_idf.npy', 'resp_embeds_idf.npy', 'word_index.pkl',
                        'conv_mask.npy', 'definitions.pkl', 'arithmetic_facts.pkl',
                        'relations_slim.pkl']:
                p = os.path.join(args.output, fn)
                if os.path.exists(p):
                    os.remove(p)

    tok = build_tokenizer(texts, args.vocab, args.output)
    V = min(tok.get_vocab_size(), args.vocab)
    embeds, idf = build_embeddings(texts, tok, V, args.dim, args.output)
    n = build_corpus_and_index(texts, tok, embeds, idf, V, args.dim, args.output)

    dt = time.time() - t_total
    print(f"\n{'='*50}")
    print(f"  Done in {dt:.0f}s ({dt/60:.1f} min)")
    print(f"  Corpus:  {n:,} entries")
    total_size = sum(os.path.getsize(os.path.join(args.output, f))
                     for f in os.listdir(args.output)
                     if os.path.isfile(os.path.join(args.output, f)))
    print(f"  Size:    {total_size/1e9:.1f} GB")
    print(f"{'='*50}")
    print(f"\n  Ready. Run: python chat.py")


if __name__ == '__main__':
    main()
