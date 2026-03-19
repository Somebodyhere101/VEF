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

    def __init__(self, embeddings, n_heads=4, n_layers=2):
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

    def _build_head(self, head_idx, E, U, S, Vt):
        """Build Q/K/V for a specialized head.

        Each head captures a different type of token relationship:
          0: Local syntax (nearby tokens matter most)
          1: Semantic similarity (meaning-based attention)
          2: Functional role (what role does this token play)
          3: Contrastive (attend to DIFFERENT tokens for discrimination)
        """
        d = self.dim
        hd = self.head_dim
        s = head_idx * hd
        e = s + hd

        if head_idx == 0:
            # SYNTACTIC HEAD: Q and K from adjacent SVD dimensions
            # Adjacent SVD dimensions capture local co-occurrence patterns
            # Q projects into the "what do I need syntactically" space
            # K projects into the "what do I offer syntactically" space
            wq = Vt[s:e, :].T.copy()
            # K uses REVERSED order of same dimensions — creates asymmetry
            # so that subject attends to verb, verb to object, etc.
            wk = Vt[s:e, :][::-1].T.copy()
            # V: direct embedding slice
            wv = (np.diag(S[s:e]) @ Vt[s:e, :]).T.copy()
            wv /= np.maximum(np.linalg.norm(wv, axis=0, keepdims=True), 1e-10)

        elif head_idx == 1:
            # SEMANTIC HEAD: Q and K emphasize high-variance dimensions
            # The top singular values capture the most important semantic axes
            # Weight by singular values to emphasize meaning
            sq = np.diag(S[s:e] / (S[s:e].max() + 1e-10))
            wq = (Vt[s:e, :].T @ sq).copy()
            # K: same weighting but different rotation
            rot = self._random_orthogonal(hd, seed=42 + head_idx)
            wk = (Vt[s:e, :].T @ sq @ rot).copy()
            # V: information-preserving projection
            wv = Vt[s:e, :].T.copy()

        elif head_idx == 2:
            # FUNCTIONAL HEAD: Q and K from cross-covariance
            # Different parts of the SVD capture different functional roles
            # Use SVD dimensions that are ORTHOGONAL to head 0 and 1
            wq = Vt[s:e, :].T.copy()
            # K: project through the ANTI-correlation structure
            # Tokens with opposite signs in certain dimensions have
            # complementary roles (e.g., modifier↔noun, operator↔operand)
            wk = -Vt[s:e, :][::-1].T.copy()
            wv = (np.diag(np.sqrt(np.abs(S[s:e]))) @ Vt[s:e, :]).T.copy()
            wv /= np.maximum(np.linalg.norm(wv, axis=0, keepdims=True), 1e-10)

        else:
            # CONTRASTIVE HEAD: attend to what's DIFFERENT
            # Q: what am I? K: what contrasts with me?
            # This creates attention that highlights novel/surprising tokens
            wq = Vt[s:e, :].T.copy()
            # K: the residual after projecting out the dominant direction
            # This makes tokens attend to what they DON'T share
            dominant = Vt[0:1, :].T  # First singular vector
            wk_full = Vt[s:e, :].T.copy()
            # Remove the dominant direction from K
            for i in range(hd):
                proj = np.dot(wk_full[:, i], dominant[:, 0]) * dominant[:, 0]
                wk_full[:, i] -= proj
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

            # Add position bias per head
            for h in range(min(self.n_heads, len(head_decays))):
                decay = head_decays[h]
                pos_bias = -rel_dist / decay  # nearby=0, far=negative
                scores[h] = scores[h] + pos_bias

            if mask is not None:
                scores = scores + mask.unsqueeze(0)

            attn = F.softmax(scores, dim=2)

            # Weighted values
            head_outs = torch.bmm(attn, V)
            concat = head_outs.permute(1, 0, 2).reshape(n, -1)

            # Residual + layer norm
            H = F.layer_norm(H + concat @ self.W_O, [self.dim])

        return H
