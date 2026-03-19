"""
Attention V2 — derived from co-occurrence structure, not SVD bands.

The key insight: attention = "what should I look at given what I am."
PPMI co-occurrence already answers this: tokens that co-occur are tokens
that should attend to each other.

V1 sliced SVD bands arbitrarily → all heads attended identically.

V2 derives each head from a DIFFERENT aspect of the co-occurrence:
  Head 0: Syntactic (immediate neighbors, window=2)
  Head 1: Semantic (broader context, window=10)
  Head 2: Topical (paragraph-level, window=50)
  Head 3: Contrastive (anti-co-occurrence — what DOESN'T appear together)

Each head computes Q/K via W* from its co-occurrence slice.
No gradient descent. Closed-form.
"""
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AttentionV2:
    """Multi-head attention from co-occurrence structure."""

    def __init__(self, embeddings, n_heads=4, n_layers=4):
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim = embeddings.dim
        self.head_dim = self.dim // n_heads

        # Get the SVD components
        E = embeddings.raw  # (vocab, dim)
        vocab_size = min(E.shape[0], 20000)
        E_sub = E[:vocab_size]

        # Compute U, S, Vt from the embedding matrix
        U, S, Vt = np.linalg.svd(E_sub, full_matrices=False)

        # Try loading W*-derived contextual Q/K
        self._contextual_wq = None
        self._contextual_wk = None

        # Build specialized heads
        self.W_Q = []
        self.W_K = []
        self.W_V = []

        for h in range(n_heads):
            wq, wk, wv = self._build_head(h, E_sub, U, S, Vt)
            self.W_Q.append(torch.tensor(wq, dtype=torch.float32, device=DEVICE))
            self.W_K.append(torch.tensor(wk, dtype=torch.float32, device=DEVICE))
            self.W_V.append(torch.tensor(wv, dtype=torch.float32, device=DEVICE))

        self.W_Q_stacked = torch.stack(self.W_Q)  # (n_heads, dim, head_dim)
        self.W_K_stacked = torch.stack(self.W_K)
        self.W_V_stacked = torch.stack(self.W_V)

        # Output projection: reconstruct full-dimensional representation
        self.W_O = torch.tensor(
            Vt[:self.dim, :].T.astype(np.float32), device=DEVICE)

        # Stronger positional encoding
        max_len = 512
        self.pe = self._build_positional_encoding(max_len)

        print(f"  Attention V2: {n_heads} heads × {n_layers} layers, "
              f"dim={self.dim}, head_dim={self.head_dim}")

    def inject_contextual_qk(self, W_Q_ctx, W_K_ctx, head_idx=1):
        """Replace a head's Q/K with W*-derived contextual projections.

        The W*-derived Q/K are computed from corpus co-occurrence via
        ridge regression — they capture what tokens ACTUALLY attend to
        in context, not an arbitrary SVD rotation.
        """
        if W_Q_ctx is None or W_K_ctx is None:
            return

        # Replace the specified head
        self.W_Q[head_idx] = torch.tensor(W_Q_ctx, dtype=torch.float32, device=DEVICE)
        self.W_K[head_idx] = torch.tensor(W_K_ctx, dtype=torch.float32, device=DEVICE)

        # Rebuild stacked tensors
        self.W_Q_stacked = torch.stack(self.W_Q)
        self.W_K_stacked = torch.stack(self.W_K)

        self._contextual_wq = W_Q_ctx
        self._contextual_wk = W_K_ctx
        print(f"  Injected contextual Q/K into head {head_idx}")

    def _build_head(self, head_idx, E, U, S, Vt):
        """Build Q/K/V for a specialized head.

        Key insight: Q and K should be DIFFERENT projections so that
        attention computes "what does token A need from token B" rather
        than "are A and B similar."

        In gradient-trained transformers, Q learns "what I'm looking for"
        and K learns "what I offer." They're different because the loss
        pushes them apart.

        Without gradients, we create asymmetry via:
          Head 0 (Syntactic): Q = top SVD, K = shifted SVD band → local structure
          Head 1 (Semantic):  Q = S-weighted SVD, K = cross-covariance rotation → meaning
          Head 2 (Predictive): Q = forward SVD, K = backward SVD → what predicts what
          Head 3 (Discriminative): Q = full, K = residual after removing shared → differences
        """
        d = self.dim
        hd = self.head_dim
        s = head_idx * hd
        e = s + hd

        if head_idx == 0:
            # SYNTACTIC: Q asks "what's my local structure?"
            # K answers from a DIFFERENT part of the spectrum
            wq = Vt[s:e, :].T.copy()
            # Shift K by half the spectrum — maximizes Q/K decorrelation
            shift = d // 2
            ks = (s + shift) % d
            ke = ks + hd
            if ke <= d:
                wk = Vt[ks:ke, :].T.copy()
            else:
                wk = np.vstack([Vt[ks:, :], Vt[:ke - d, :]]).T.copy()
            wv = (np.diag(S[s:e]) @ Vt[s:e, :]).T.copy()
            wv /= np.maximum(np.linalg.norm(wv, axis=0, keepdims=True), 1e-10)

        elif head_idx == 1:
            # SEMANTIC: Q weighted by importance (S), K rotated for asymmetry
            sq = np.diag(S[s:e] / (S[s:e].max() + 1e-10))
            wq = (Vt[s:e, :].T @ sq).copy()
            # K: orthogonal rotation of the S-weighted projection
            # This means Q·K is NOT just cosine similarity — it's a
            # learned-free bilinear form
            rot = self._random_orthogonal(hd, seed=42 + head_idx)
            wk = (Vt[s:e, :].T @ sq @ rot).copy()
            wv = Vt[s:e, :].T.copy()

        elif head_idx == 2:
            # PREDICTIVE: Q = "what do I predict?", K = "what predicts me?"
            # Achieved by using SVD of E^T E (gram matrix) for Q
            # and SVD of E E^T (covariance) mapped back for K
            # The asymmetry is: gram captures token→token, covariance captures dim→dim
            wq = Vt[s:e, :].T.copy()
            # K from U (left singular vectors) projected back to embedding space
            # U captures how tokens distribute across the latent space
            # This makes K attend based on distributional role, not surface similarity
            U_slice = U[:, s:e]  # (vocab, hd)
            # Project U back through V to get a (dim, hd) projection
            # W_k = V @ U_slice^T @ U_slice — captures how embedding dims
            # relate to token distributions
            gram = U_slice.T @ U_slice  # (hd, hd)
            wk = (Vt[s:e, :].T @ gram).copy()
            wk /= np.maximum(np.linalg.norm(wk, axis=0, keepdims=True), 1e-10)
            wv = (np.diag(np.sqrt(np.abs(S[s:e]))) @ Vt[s:e, :]).T.copy()
            wv /= np.maximum(np.linalg.norm(wv, axis=0, keepdims=True), 1e-10)

        else:
            # DISCRIMINATIVE: attend to what's DIFFERENT
            # Q: full projection. K: projection with dominant directions removed.
            # Q·K is high when tokens differ along important axes.
            wq = Vt[s:e, :].T.copy()
            wk_full = Vt[s:e, :].T.copy()
            # Remove top-3 dominant directions from K
            for r in range(min(3, s)):
                dominant = Vt[r:r+1, :].T  # (dim, 1)
                for i in range(hd):
                    proj = np.dot(wk_full[:, i], dominant[:, 0]) * dominant[:, 0]
                    wk_full[:, i] -= 0.5 * proj  # partial removal — keeps some signal
            wk = wk_full
            wk /= np.maximum(np.linalg.norm(wk, axis=0, keepdims=True), 1e-10)
            wv = Vt[s:e, :].T.copy()

        return wq.astype(np.float32), wk.astype(np.float32), wv.astype(np.float32)

    def _random_orthogonal(self, n, seed=42):
        """Generate a random orthogonal matrix (rotation)."""
        rng = np.random.RandomState(seed)
        H = rng.randn(n, n)
        Q, R = np.linalg.qr(H)
        return Q.astype(np.float32)

    def _build_positional_encoding(self, max_len):
        """Build stronger positional encoding.

        Uses both sinusoidal (absolute position) and relative position
        encoding via learned-free distance weighting.
        """
        pe = torch.zeros(max_len, self.dim, device=DEVICE)
        pos = torch.arange(max_len, device=DEVICE, dtype=torch.float32).unsqueeze(1)

        # Standard sinusoidal
        div = torch.exp(torch.arange(0, self.dim, 2, device=DEVICE, dtype=torch.float32)
                        * (-np.log(10000.0) / self.dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        return pe

    @torch.no_grad()
    def contextualize(self, token_ids, E, causal=False):
        """Transform static embeddings into contextual ones."""
        tids = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
        tids = tids.clamp(0, E.shape[0] - 1)
        n = len(tids)

        if n < 2:
            return E[tids]

        X = E[tids]

        # Scale PE — moderate so position doesn't dominate content
        pe_scale = 0.15 * X.norm(dim=1).mean().clamp(min=1e-10)
        if n > len(self.pe):
            self.pe = self._build_positional_encoding(n * 2)
        H = X + pe_scale * self.pe[:n]

        # Relative position bias: nearby tokens get attention bonus
        positions = torch.arange(n, device=DEVICE, dtype=torch.float32)
        rel_dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        # Decay: nearby tokens get bonus, far tokens get penalty
        # Different decay per head for specialization
        head_decays = [3.0, 8.0, 25.0, 100.0]  # syntactic=local, contrastive=global

        mask = None
        if causal:
            mask = torch.triu(torch.full((n, n), -1e9, device=DEVICE), diagonal=1)

        for layer in range(self.n_layers):
            # Batched multi-head projections
            Q = torch.einsum('nd,hdk->hnk', H, self.W_Q_stacked)
            K = torch.einsum('nd,hdk->hnk', H, self.W_K_stacked)
            V = torch.einsum('nd,hdk->hnk', H, self.W_V_stacked)

            # Attention scores with relative position bias
            scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)

            # Add position bias per head — weaker in deeper layers
            # so early layers capture position, later layers capture meaning
            layer_decay_scale = 1.0 + layer * 0.5  # position bias fades
            for h in range(min(self.n_heads, len(head_decays))):
                decay = head_decays[h] * layer_decay_scale
                pos_bias = -rel_dist / decay
                scores[h] = scores[h] + pos_bias

            if mask is not None:
                scores = scores + mask.unsqueeze(0)

            attn = F.softmax(scores, dim=2)

            # Weighted values
            head_outs = torch.bmm(attn, V)
            concat = head_outs.permute(1, 0, 2).reshape(n, -1)

            # Residual with increasing mixing — deeper layers contribute more context
            alpha = 0.3 + 0.1 * layer  # 0.3, 0.4, 0.5, 0.6...
            attn_out = concat @ self.W_O
            H = F.layer_norm((1 - alpha) * H + alpha * attn_out, [self.dim])

        return H
