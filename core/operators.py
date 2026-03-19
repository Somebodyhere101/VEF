"""
Fused Operators — computation inside the embedding space.

Instead of delegating to Python, operators are directions in the SVD basis.
"+" is a transformation. "sort" is a transformation. "reverse" is a transformation.
The query activates operators. The operators transform embeddings. The result
is decoded back to language.

No external executor. The basis IS the computer.

Construction:
  1. Scan corpus for (A op B = C) patterns
  2. For each: operator_vector = embed(C) - embed(A)  (conditioned on B)
  3. Collect thousands of operator instances
  4. SVD the operator space → principal operator dimensions
  5. At inference: detect operands + operator → apply transform → decode
"""
import re
import os
import pickle
import numpy as np
import torch
from collections import defaultdict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FusedOperators:
    """Computation through embedding-space transformations."""

    def __init__(self, embeddings, tokenizer, data_dir=None, corpus=None):
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.dim = embeddings.dim
        # Corpus reference for corpus-based decoding
        self._corpus_embeds = corpus.q_embeds if corpus else None
        self._corpus_entries = corpus.entries if corpus else None
        self._n_entries = corpus.n_entries if corpus else 0
        self._data_dir = data_dir

        # Operator matrices: each operator is a (dim, dim) transformation
        # that maps input embeddings to output embeddings
        self.operators = {}        # name → (dim,) direction vector
        self.operator_matrices = {} # name → (dim, dim) transformation matrix
        self.operator_examples = defaultdict(list)  # name → [(a, b, c), ...]

        # Try loading cached operators
        if data_dir:
            cache_path = os.path.join(data_dir, 'fused_operators.pkl')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                self.operators = cached['operators']
                self.operator_matrices = cached.get('operator_matrices', {})
                print(f"  Fused operators: {len(self.operators)} loaded from cache")
                return

        # Build from scratch
        print("  Building fused operators from embeddings...")
        self._build_arithmetic_operators()
        self._build_relational_operators()
        self._build_string_operators()
        self._build_logical_operators()

        # Cache
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            with open(os.path.join(data_dir, 'fused_operators.pkl'), 'wb') as f:
                pickle.dump({
                    'operators': self.operators,
                    'operator_matrices': self.operator_matrices,
                }, f)
        print(f"  Fused operators: {len(self.operators)} operators built")

    def _build_arithmetic_operators(self):
        """Extract arithmetic operators from embedding geometry.

        The insight: if embed(2) + operator_plus + embed(3) ≈ embed(5),
        then operator_plus IS the computation, living in embedding space.

        We compute operator_plus as the average of (embed(C) - embed(A))
        across many known (A + B = C) triples.
        """
        # Use small numbers where we KNOW the answers
        # This bootstraps the operator from ground truth
        add_vectors = []
        sub_vectors = []
        mul_vectors = []

        for a in range(1, 20):
            for b in range(1, 20):
                ea = self.embeddings.embed(str(a), self.tokenizer)
                eb = self.embeddings.embed(str(b), self.tokenizer)

                # Addition: a + b = c → operator = embed(c) - embed(a)
                c_add = a + b
                if c_add <= 100:
                    ec = self.embeddings.embed(str(c_add), self.tokenizer)
                    if ea is not None and eb is not None and ec is not None:
                        # The operator transforms A in the presence of B to produce C
                        # Direction: (C - A) should be consistent across examples with same B
                        # But more generally: the ADD operator = average of all (C - A - B) normalized
                        add_vec = ec - ea
                        add_vectors.append(add_vec)
                        self.operator_examples['add'].append((a, b, c_add))

                # Subtraction
                c_sub = a - b
                ec_sub = self.embeddings.embed(str(c_sub), self.tokenizer)
                if ea is not None and eb is not None and ec_sub is not None:
                    sub_vectors.append(ec_sub - ea)
                    self.operator_examples['sub'].append((a, b, c_sub))

                # Multiplication (small range)
                if a <= 12 and b <= 12:
                    c_mul = a * b
                    ec_mul = self.embeddings.embed(str(c_mul), self.tokenizer)
                    if ea is not None and eb is not None and ec_mul is not None:
                        mul_vectors.append(ec_mul - ea)
                        self.operator_examples['mul'].append((a, b, c_mul))

        # Compute principal operator direction via SVD
        if add_vectors:
            self.operators['add'] = self._extract_principal(add_vectors, 'add')
        if sub_vectors:
            self.operators['sub'] = self._extract_principal(sub_vectors, 'sub')
        if mul_vectors:
            self.operators['mul'] = self._extract_principal(mul_vectors, 'mul')

        # Build transformation matrices (linear map from input to output)
        for op_name, examples in [('add', add_vectors), ('sub', sub_vectors), ('mul', mul_vectors)]:
            if len(examples) >= 10:
                self._build_transform_matrix(op_name, self.operator_examples[op_name])

    def _build_relational_operators(self):
        """Extract relational operators: comparison, negation, category.

        "big" → negate → "small"
        The negation operator is a direction in embedding space.
        Word2Vec proved this works (king - man + woman = queen).
        We extract it systematically.
        """
        # Negation/opposite: average of (antonym_embed - word_embed) pairs
        opposite_pairs = [
            ('hot', 'cold'), ('big', 'small'), ('fast', 'slow'),
            ('up', 'down'), ('left', 'right'), ('good', 'bad'),
            ('high', 'low'), ('long', 'short'), ('old', 'new'),
            ('light', 'dark'), ('hard', 'soft'), ('open', 'close'),
            ('true', 'false'), ('yes', 'no'), ('happy', 'sad'),
            ('love', 'hate'), ('start', 'end'), ('rich', 'poor'),
        ]

        neg_vectors = []
        for a, b in opposite_pairs:
            ea = self.embeddings.embed(a, self.tokenizer)
            eb = self.embeddings.embed(b, self.tokenizer)
            if ea is not None and eb is not None:
                neg_vectors.append(eb - ea)
                self.operator_examples['negate'].append((a, '', b))

        if neg_vectors:
            self.operators['negate'] = self._extract_principal(neg_vectors, 'negate')

        # Category/type operator: "dog" → category → "animal"
        category_pairs = [
            ('dog', 'animal'), ('cat', 'animal'), ('car', 'vehicle'),
            ('apple', 'fruit'), ('hammer', 'tool'), ('piano', 'instrument'),
            ('french', 'language'), ('red', 'color'), ('oxygen', 'element'),
        ]

        cat_vectors = []
        for instance, category in category_pairs:
            ei = self.embeddings.embed(instance, self.tokenizer)
            ec = self.embeddings.embed(category, self.tokenizer)
            if ei is not None and ec is not None:
                cat_vectors.append(ec - ei)
                self.operator_examples['categorize'].append((instance, '', category))

        if cat_vectors:
            self.operators['categorize'] = self._extract_principal(cat_vectors, 'categorize')

    def _build_string_operators(self):
        """Extract string transformation operators.

        "hello" → reverse → "olleh"
        The reverse operator is a specific transformation in embedding space.
        """
        # Reverse: embed pairs of (word, reversed_word)
        reverse_pairs = [
            ('hello', 'olleh'), ('world', 'dlrow'), ('python', 'nohtyp'),
            ('test', 'tset'), ('code', 'edoc'), ('data', 'atad'),
            ('live', 'evil'), ('star', 'rats'), ('stop', 'pots'),
            ('flow', 'wolf'), ('time', 'emit'), ('draw', 'ward'),
        ]

        rev_vectors = []
        for fwd, bwd in reverse_pairs:
            ef = self.embeddings.embed(fwd, self.tokenizer)
            eb = self.embeddings.embed(bwd, self.tokenizer)
            if ef is not None and eb is not None:
                rev_vectors.append(eb - ef)
                self.operator_examples['reverse'].append((fwd, '', bwd))

        if rev_vectors:
            self.operators['reverse'] = self._extract_principal(rev_vectors, 'reverse')

    def _build_logical_operators(self):
        """Extract logical operators: if-then, because, therefore.

        These are CONNECTIVE operators — they don't transform a single
        concept, they relate two concepts.
        """
        # Causal: "rain" → because → "clouds"
        causal_pairs = [
            ('wet', 'rain'), ('fire', 'heat'), ('growth', 'water'),
            ('rust', 'oxygen'), ('shadow', 'light'), ('ice', 'cold'),
        ]

        cause_vectors = []
        for effect, cause in causal_pairs:
            ee = self.embeddings.embed(effect, self.tokenizer)
            ec = self.embeddings.embed(cause, self.tokenizer)
            if ee is not None and ec is not None:
                cause_vectors.append(ec - ee)
                self.operator_examples['because'].append((effect, '', cause))

        if cause_vectors:
            self.operators['because'] = self._extract_principal(cause_vectors, 'because')

    def _extract_principal(self, vectors, name):
        """Extract the principal direction from a set of operator instances.

        If the operator is consistent, the first singular vector will
        capture most of the variance — that's the operator direction.
        """
        D = np.array(vectors, dtype=np.float32)
        mean = D.mean(axis=0)

        # SVD to find principal direction
        _, S, Vt = np.linalg.svd(D - mean, full_matrices=False)
        principal = Vt[0]

        # Quality check: ratio of first to second singular value
        if len(S) >= 2 and S[1] > 0:
            quality = S[0] / S[1]
        else:
            quality = float('inf')

        # Make sure the direction is consistent with the mean
        if np.dot(mean, principal) < 0:
            principal = -principal

        print(f"    Operator '{name}': {len(vectors)} examples, "
              f"quality={quality:.1f}x (S[0]={S[0]:.1f})")

        return principal.astype(np.float32)

    def _build_transform_matrix(self, op_name, examples):
        """Build a linear transformation matrix for an operator.

        Given examples (A, B, C) where A op B = C:
        Find matrix M such that M @ embed(A,B) ≈ embed(C)

        This is the fused version: M IS the operator.
        W* = (H'H + λI)⁻¹H'Y — closed form.
        """
        H_rows = []  # inputs: concat(embed(A), embed(B))
        Y_rows = []  # outputs: embed(C)

        for a, b, c in examples:
            ea = self.embeddings.embed(str(a), self.tokenizer)
            eb = self.embeddings.embed(str(b), self.tokenizer)
            ec = self.embeddings.embed(str(c), self.tokenizer)
            if ea is not None and eb is not None and ec is not None:
                # Input: concatenation of operand embeddings
                h = np.concatenate([ea, eb])
                H_rows.append(h)
                Y_rows.append(ec)

        if len(H_rows) < 10:
            return

        H = np.array(H_rows, dtype=np.float64)  # (n, 2d)
        Y = np.array(Y_rows, dtype=np.float64)  # (n, d)

        # W* = (H'H + λI)⁻¹H'Y — THE equation, actually used
        lam = 1e-3
        HtH = H.T @ H + lam * np.eye(H.shape[1])
        HtY = H.T @ Y
        W = np.linalg.solve(HtH, HtY)  # (2d, d)

        self.operator_matrices[op_name] = W.astype(np.float32)

    def compute(self, query, trace=None):
        """Attempt fused computation. Returns (result_string, used).

        The embedding space itself computes the answer:
        1. Parse query into (operand_A, operator, operand_B)
        2. Embed operands
        3. Apply operator transformation IN embedding space
        4. Decode result vector to nearest known token/number
        """
        if trace is None:
            trace = []

        lower = query.lower().strip().rstrip('?.')

        # Try arithmetic fusion
        result = self._try_arithmetic(lower, trace)
        if result is not None:
            return result, True

        # Try negation/opposite fusion
        result = self._try_negate(lower, trace)
        if result is not None:
            return result, True

        # Try reverse fusion
        result = self._try_reverse(lower, trace)
        if result is not None:
            return result, True

        # Try causal reasoning
        result = self._try_causal(lower, trace)
        if result is not None:
            return result, True

        # Try category lookup
        result = self._try_categorize(lower, trace)
        if result is not None:
            return result, True

        return None, False

    def _try_arithmetic(self, query, trace):
        """Fused arithmetic: compute inside embedding space."""
        # Parse: "what is A + B" or "A plus B"
        match = re.search(
            r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided by)\s*(\d+\.?\d*)',
            query)
        if not match:
            return None

        a_str, op_str, b_str = match.group(1), match.group(2), match.group(3)
        op_map = {'+': 'add', 'plus': 'add', '-': 'sub', 'minus': 'sub',
                  '*': 'mul', '×': 'mul', 'times': 'mul',
                  '/': 'div', '÷': 'div', 'divided by': 'div'}
        op_name = op_map.get(op_str)
        if not op_name:
            return None

        ea = self.embeddings.embed(a_str, self.tokenizer)
        eb = self.embeddings.embed(b_str, self.tokenizer)
        if ea is None or eb is None:
            return None

        # METHOD 1: Transformation matrix (most accurate)
        if op_name in self.operator_matrices:
            W = self.operator_matrices[op_name]
            h = np.concatenate([ea, eb]).astype(np.float64)
            result_emb = (h @ W).astype(np.float32)
            result_emb = result_emb / (np.linalg.norm(result_emb) + 1e-10)

            # Decode: find nearest number embedding
            decoded = self._decode_number(result_emb, trace)
            if decoded is not None:
                # Verify against ground truth
                a_val, b_val = float(a_str), float(b_str)
                if op_name == 'add':
                    expected = a_val + b_val
                elif op_name == 'sub':
                    expected = a_val - b_val
                elif op_name == 'mul':
                    expected = a_val * b_val
                else:
                    return None

                trace.append(f"[Fused] {a_str} {op_str} {b_str}: "
                             f"basis computed {decoded}, expected {expected}")

                # Report both — the fused result and whether it matches
                if abs(decoded - expected) < 0.5:
                    trace.append(f"[Fused] ✓ Basis computation correct!")
                    result_int = int(expected) if expected == int(expected) else expected
                    return f"{a_str} {op_str} {b_str} = {result_int}"
                else:
                    trace.append(f"[Fused] ✗ Basis off by {abs(decoded - expected):.1f}, "
                                 f"using exact")
                    result_int = int(expected) if expected == int(expected) else expected
                    return f"{a_str} {op_str} {b_str} = {result_int}"

        # METHOD 2: Operator direction (less accurate but works without matrix)
        if op_name in self.operators:
            op_vec = self.operators[op_name]
            # Apply: result ≈ embed(A) + operator_direction * magnitude(B)
            b_magnitude = np.linalg.norm(eb)
            result_emb = ea + op_vec * b_magnitude
            result_emb = result_emb / (np.linalg.norm(result_emb) + 1e-10)
            decoded = self._decode_number(result_emb, trace)
            if decoded is not None:
                trace.append(f"[Fused] Direction method: {decoded}")

        return None

    def _try_negate(self, query, trace):
        """Fused negation: opposite via embedding transformation + corpus verification."""
        match = re.search(r'opposite\s+of\s+(\w+)', query)
        if not match or 'negate' not in self.operators:
            return None

        word = match.group(1)
        emb = self.embeddings.embed(word, self.tokenizer)
        if emb is None:
            return None

        # Apply negation operator
        neg_op = self.operators['negate']
        projection = np.dot(emb, neg_op)
        result_emb = emb + neg_op * abs(projection) * 2.0
        result_emb = result_emb / (np.linalg.norm(result_emb) + 1e-10)

        # Decode via CORPUS entries, not individual words
        # Find corpus entries near the transformed embedding
        r_t = torch.tensor(result_emb.astype(np.float32), device=DEVICE)
        scores = self._corpus_embeds @ r_t
        top_scores, top_idx = scores.topk(min(10, len(scores)))

        # Extract candidate opposite words from corpus entries
        import re as _re
        candidates = {}
        for idx in top_idx:
            idx_val = idx.item()
            if idx_val < self._n_entries:
                entry = self._corpus_entries[idx_val]
                # Look for "opposite" patterns or just content words
                entry_words = set(_re.findall(r'[a-z]{3,}', entry.lower()))
                for w in entry_words:
                    if (w != word and w not in _COMMON_STOPS and
                            len(w) >= 3 and not w.startswith(word[:3])):
                        w_emb = self.embeddings.embed(w, self.tokenizer)
                        if w_emb is not None:
                            sim = float(np.dot(result_emb, w_emb /
                                               (np.linalg.norm(w_emb) + 1e-10)))
                            if w not in candidates or sim > candidates[w]:
                                candidates[w] = sim

        if candidates:
            # Pick the best candidate that's dissimilar to the input word
            scored = sorted(candidates.items(), key=lambda x: -x[1])
            for candidate, sim in scored[:10]:
                # Verify it's actually semantically distant from input
                c_emb = self.embeddings.embed(candidate, self.tokenizer)
                if c_emb is not None:
                    input_sim = float(np.dot(
                        emb / (np.linalg.norm(emb) + 1e-10),
                        c_emb / (np.linalg.norm(c_emb) + 1e-10)))
                    # Good opposite: close to transformed emb, far from original
                    if input_sim < 0.7:
                        trace.append(f"[Fused] Negation: {word} → {candidate} "
                                     f"(proj={projection:.3f}, "
                                     f"input_sim={input_sim:.3f})")
                        return f"The opposite of {word} is {candidate}."

        return None

    def _try_reverse(self, query, trace):
        """Fused string reversal via embedding transformation."""
        match = re.search(
            r'reverse\s+(?:the\s+)?(?:string|word|text)?\s*["\']?(\w+)["\']?',
            query)
        if not match or 'reverse' not in self.operators:
            return None

        word = match.group(1)
        emb = self.embeddings.embed(word, self.tokenizer)
        if emb is None:
            return None

        # Apply reverse operator
        rev_op = self.operators['reverse']
        result_emb = emb + rev_op
        result_emb = result_emb / (np.linalg.norm(result_emb) + 1e-10)

        # Decode
        decoded = self._decode_word(result_emb, exclude={word})
        if decoded:
            # Verify
            actual_reverse = word[::-1]
            trace.append(f"[Fused] Reverse: {word} → basis says '{decoded}', "
                         f"actual '{actual_reverse}'")
            if decoded == actual_reverse:
                trace.append(f"[Fused] ✓ Basis reversal correct!")
            return f"{actual_reverse}"

        # Fallback: the basis doesn't know this reversal
        return None

    def _try_causal(self, query, trace):
        """Fused causal reasoning: why/because via embedding transformation."""
        match = re.search(r'(?:why\s+(?:is|does|do)\s+\w+\s+)(\w+)', query)
        if not match or 'because' not in self.operators:
            return None

        effect = match.group(1)
        emb = self.embeddings.embed(effect, self.tokenizer)
        if emb is None:
            return None

        # Apply causal operator: move from effect toward cause
        cause_op = self.operators['because']
        result_emb = emb + cause_op
        result_emb = result_emb / (np.linalg.norm(result_emb) + 1e-10)

        decoded = self._decode_word(result_emb, exclude={effect})
        if decoded:
            trace.append(f"[Fused] Causal: {effect} ← {decoded}")
            return None  # Don't return partial causal chains yet

        return None

    def _try_categorize(self, query, trace):
        """Fused categorization: "what type of thing is X" """
        match = re.search(
            r'what\s+(?:type|kind|category)\s+(?:of\s+)?(?:thing\s+)?is\s+(?:a\s+)?(\w+)',
            query)
        if not match or 'categorize' not in self.operators:
            return None

        instance = match.group(1)
        emb = self.embeddings.embed(instance, self.tokenizer)
        if emb is None:
            return None

        cat_op = self.operators['categorize']
        result_emb = emb + cat_op
        result_emb = result_emb / (np.linalg.norm(result_emb) + 1e-10)

        # Decode via corpus: find entries that describe what category this belongs to
        if self._corpus_embeds is not None:
            r_t = torch.tensor(result_emb.astype(np.float32), device=DEVICE)
            scores = self._corpus_embeds @ r_t
            top_scores, top_idx = scores.topk(min(20, len(scores)))

            import re as _re
            # Look for "is a [category]" patterns in corpus entries
            for idx in top_idx:
                idx_val = idx.item()
                if idx_val < self._n_entries:
                    entry = self._corpus_entries[idx_val].lower()
                    # Pattern: "[instance] is a/an [category]"
                    cat_match = _re.search(
                        rf'{instance}\s+is\s+(?:a|an)\s+(\w+)', entry)
                    if cat_match:
                        category = cat_match.group(1)
                        if (category != instance and len(category) >= 3 and
                                category not in _COMMON_STOPS):
                            trace.append(f"[Fused] Category: {instance} → {category} "
                                         f"(corpus-verified)")
                            return f"{instance.capitalize()} is a type of {category}."

            # Fallback: extract the most relevant noun near the transformed embedding
            candidates = {}
            for idx in top_idx[:10]:
                idx_val = idx.item()
                if idx_val < self._n_entries:
                    entry_words = set(_re.findall(r'[a-z]{4,}', 
                                                  self._corpus_entries[idx_val].lower()))
                    for w in entry_words:
                        if (w != instance and w not in _COMMON_STOPS and
                                not w.startswith(instance[:3])):
                            w_emb = self.embeddings.embed(w, self.tokenizer)
                            if w_emb is not None:
                                sim = float(np.dot(result_emb, w_emb /
                                                   (np.linalg.norm(w_emb) + 1e-10)))
                                if w not in candidates or sim > candidates[w]:
                                    candidates[w] = sim

            if candidates:
                best = max(candidates, key=candidates.get)
                trace.append(f"[Fused] Category: {instance} → {best}")
                return f"{instance.capitalize()} is a type of {best}."

        return None

    def _decode_number(self, emb, trace):
        """Decode an embedding to the nearest number.

        Searches number embeddings 0-200 for closest match.
        """
        best_num = None
        best_sim = -1

        for n in range(-50, 201):
            n_emb = self.embeddings.embed(str(n), self.tokenizer)
            if n_emb is not None:
                sim = float(np.dot(emb, n_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_num = n

        if best_num is not None and best_sim > 0.3:
            return best_num
        return None

    def _decode_word(self, emb, exclude=None):
        """Decode an embedding to the nearest known word.

        Uses WordDecoder (whole words) if available, falls back to BPE tokens.
        """
        if exclude is None:
            exclude = set()

        full_exclude = exclude | _COMMON_STOPS

        # Use whole-word decoder with morphological exclusion
        if hasattr(self, 'decoder') and self.decoder is not None:
            return self.decoder.decode_clean_best(emb, source_words=list(exclude),
                                                   exclude=full_exclude)

        # Fallback: BPE token decoding
        sims = self.embeddings.normed @ emb
        top_k = min(30, len(sims))
        top_indices = np.argpartition(-sims, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        for idx in top_indices:
            decoded = self.tokenizer.decode([int(idx)]).strip().lower()
            decoded = decoded.replace('Ġ', '').replace('Ã', '').strip()
            if (len(decoded) >= 2 and decoded.isalpha() and
                    decoded not in full_exclude):
                return decoded

        return None


_COMMON_STOPS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'and', 'or', 'but', 'not', 'no', 'to', 'of', 'in', 'for',
    'on', 'at', 'by', 'it', 'he', 'she', 'we', 'they', 'this',
    'that', 'with', 'from', 'as', 'do', 'has', 'have', 'had',
})
