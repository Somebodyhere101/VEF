"""
Boundary-Driven Composition — the medium timescale.

When a query hits the boundary region (partial basis energy, high concentration),
the system can't just retrieve — no single corpus entry covers it. Instead:

1. Decompose the query into basis components (what IS understood)
2. Measure the anti-basis residual (what's missing)
3. Retrieve multiple entries that each cover different components
4. Compose their responses, weighted by how well they fill the gap

This is how novel answers emerge from frozen statistics.
"""
import re
import numpy as np
import torch
import json
import os
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BoundaryComposer:
    """Generate novel responses by composing across the basis boundary."""

    def __init__(self, embeddings, corpus, tokenizer, awareness, retrieval):
        self.embeddings = embeddings
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.awareness = awareness
        self.retrieval = retrieval

        # Boundary log for slow-timescale learning
        self._boundary_log_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'boundary_log.jsonl')

    def try_compose(self, query, trace=None):
        """Attempt boundary-driven composition. Returns (response, used) or (None, False).

        Only activates when the query is in the boundary region —
        partial basis energy, too novel for pure retrieval.
        """
        if trace is None:
            trace = []

        # Step 1: Measure where the query falls
        q_emb = self.embeddings.embed(query, self.tokenizer)
        if q_emb is None:
            return None, False

        # Check each concept word's awareness status
        words = [w for w in re.findall(r'[a-z]{3,}', query.lower())
                 if w not in _STOP_WORDS]
        if len(words) < 2:
            return None, False  # Need multiple concepts to compose

        basis_words = []    # Words the model knows (meaning OR partial with high energy)
        boundary_words = [] # Words partially known (boundary region)
        unknown_words = []  # Words fully in anti-basis

        for w in words:
            cat, energy, conc, _ = self.awareness.measure(w)
            if cat == 'meaning' or (cat == 'partial' and energy > 0.85):
                basis_words.append((w, energy, conc))
            elif cat == 'partial':
                boundary_words.append((w, energy, conc))
            else:
                unknown_words.append((w, energy, conc))

        # Need at least some basis words to compose from
        # Partial words count as basis if they have high enough energy
        if not basis_words and not boundary_words:
            return None, False
        # If no basis but some boundary, treat boundary as basis
        if not basis_words:
            basis_words = boundary_words[:2]
            boundary_words = boundary_words[2:]

        n_known = len(basis_words)
        n_total = len(words)
        boundary_ratio = 1.0 - (n_known / n_total)

        # Check if the combination is novel even if individual words are known
        combination_novel = False
        if n_known >= 2 and not boundary_words and not unknown_words:
            # All words known individually — but does the corpus have them TOGETHER?
            q_emb_full = self.embeddings.embed(query, self.tokenizer)
            if q_emb_full is not None:
                q_t = torch.tensor(q_emb_full, dtype=torch.float32, device=DEVICE)
                best_score = float((self.corpus.q_embeds @ q_t).max())
                # If best corpus match is weak despite all words being known,
                # the COMBINATION is novel — compose!
                if best_score < 0.75:
                    combination_novel = True
                    boundary_ratio = 0.5  # Force composition
                    trace.append(f"[Boundary] Novel combination detected "
                                 f"(best corpus match={best_score:.3f})")

        if not combination_novel and (not boundary_words and not unknown_words):
            return None, False

        if boundary_ratio < 0.2:
            return None, False  # Mostly known, retrieval is fine

        trace.append(f"[Boundary] {n_known}/{n_total} concepts in basis, "
                     f"composing across boundary (ratio={boundary_ratio:.2f})")

        # Step 2: Decompose query into basis projections
        # Project query embedding onto principal components
        basis_projection = self._project_to_basis(q_emb)
        residual = q_emb - basis_projection
        residual_norm = float(np.linalg.norm(residual))
        basis_norm = float(np.linalg.norm(basis_projection))

        trace.append(f"[Boundary] Basis energy={basis_norm:.3f}, "
                     f"residual={residual_norm:.3f}")

        # Step 3: Multi-concept retrieval
        # Retrieve separately for each known concept, then for concept PAIRS
        fragments = []

        # Individual concept retrievals
        for w, energy, _ in basis_words[:3]:
            results = self.retrieval.search(w, top_k=5)
            for score, resp in results[:2]:
                if len(resp) > 20:
                    fragments.append((score * energy, resp, w))

        # Pair retrievals — combine known concepts
        for i, (w1, e1, _) in enumerate(basis_words[:3]):
            for w2, e2, _ in basis_words[i+1:3]:
                pair_query = f"{w1} {w2}"
                results = self.retrieval.search(pair_query, top_k=5)
                for score, resp in results[:2]:
                    if len(resp) > 20:
                        # Bonus for covering multiple concepts
                        coverage = sum(1 for w, _, _ in basis_words
                                       if w in resp.lower())
                        bonus = 1.0 + 0.3 * coverage
                        fragments.append((score * bonus, resp, pair_query))

        # Full query retrieval as baseline
        full_results = self.retrieval.search(query, top_k=5)
        for score, resp in full_results[:3]:
            if len(resp) > 20:
                fragments.append((score, resp, 'full'))

        if not fragments:
            self._log_boundary(query, basis_words, boundary_words, unknown_words)
            return None, False

        # Step 4: Score fragments by how well they fill the gap
        scored = []
        for score, resp, source in fragments:
            r_emb = self.embeddings.embed(resp[:200], self.tokenizer)
            if r_emb is None:
                continue

            # How much of the query does this fragment cover?
            query_alignment = float(np.dot(q_emb, r_emb))

            # How much of the RESIDUAL does this fragment address?
            # (This is the key innovation — prefer fragments that cover
            # what the basis alone doesn't)
            if residual_norm > 0.01:
                residual_alignment = float(np.dot(residual / residual_norm, r_emb))
            else:
                residual_alignment = 0.0

            # Combined: base relevance + residual coverage
            combined = query_alignment + 0.5 * max(0, residual_alignment)
            scored.append((combined, resp, source, r_emb))

        if not scored:
            self._log_boundary(query, basis_words, boundary_words, unknown_words)
            return None, False

        scored.sort(key=lambda x: -x[0])

        # Step 5: Compose — take top fragments and merge non-redundantly
        composed = self._merge_fragments(scored[:5], q_emb, trace)

        if composed and len(composed) > 30:
            trace.append(f"[Boundary] Composed response from {min(5, len(scored))} fragments")
            self._log_boundary(query, basis_words, boundary_words, unknown_words,
                               success=True)
            return composed, True

        self._log_boundary(query, basis_words, boundary_words, unknown_words)
        return None, False

    def _project_to_basis(self, emb):
        """Project embedding onto the corpus basis (principal subspace).

        The corpus embeddings span the 'known' subspace. Projection onto
        this subspace gives the basis component; the residual is anti-basis.
        """
        # Use top-k corpus embeddings as basis vectors (efficient approximation)
        q_t = torch.tensor(emb, dtype=torch.float32, device=DEVICE)
        scores = self.corpus.q_embeds @ q_t
        top_k = min(50, len(scores))
        _, top_idx = scores.topk(top_k)

        # Project onto subspace spanned by top matches
        basis_vecs = self.corpus.q_embeds[top_idx].cpu().numpy()  # (k, d)

        # Gram-Schmidt-ish: project onto the span
        # Simple approximation: weighted average of top matches
        weights = scores[top_idx].cpu().numpy()
        weights = np.maximum(weights, 0)
        total = weights.sum()
        if total < 1e-10:
            return np.zeros_like(emb)

        projection = (basis_vecs * weights[:, None]).sum(0) / total
        norm = np.linalg.norm(projection)
        if norm > 1e-10:
            # Scale to match original embedding magnitude in the basis direction
            projection = projection / norm * float(np.dot(emb, projection / norm))

        return projection

    def _merge_fragments(self, scored_fragments, q_emb, trace):
        """Merge top fragments into a coherent composed response.

        Strategy: take the best fragment as the base. For each additional
        fragment, extract sentences that cover concepts NOT in the base.
        Append non-redundant sentences.
        """
        if not scored_fragments:
            return None

        base_score, base_resp, base_source, base_emb = scored_fragments[0]
        composed_parts = [base_resp]
        covered_concepts = set(re.findall(r'[a-z]{3,}', base_resp.lower()))

        for score, resp, source, r_emb in scored_fragments[1:]:
            # Check if this fragment adds new concepts
            resp_concepts = set(re.findall(r'[a-z]{3,}', resp.lower()))
            new_concepts = resp_concepts - covered_concepts - _STOP_WORDS

            if len(new_concepts) < 2:
                continue  # Too redundant

            # Extract the most relevant sentence(s) from this fragment
            sentences = [s.strip() for s in re.split(r'[.!?]+', resp) if len(s.strip()) > 15]
            for sent in sentences:
                sent_concepts = set(re.findall(r'[a-z]{3,}', sent.lower()))
                sent_new = sent_concepts & new_concepts
                if len(sent_new) >= 1:
                    # Check it's not too similar to what we have
                    s_emb = self.embeddings.embed(sent, self.tokenizer)
                    if s_emb is not None:
                        similarity = float(np.dot(base_emb, s_emb))
                        if similarity < 0.85:  # Not redundant
                            composed_parts.append(sent.strip())
                            covered_concepts.update(sent_concepts)
                            trace.append(f"[Boundary] +fragment from '{source}': "
                                         f"+{len(sent_new)} concepts")
                            break  # One sentence per fragment

            if len(composed_parts) >= 4:
                break  # Don't over-compose

        # Join with proper punctuation
        result = '. '.join(p.rstrip('.!? ') for p in composed_parts if p) + '.'
        return result

    def _log_boundary(self, query, basis_words, boundary_words, unknown_words,
                      success=False):
        """Log boundary queries for slow-timescale learning.

        These are the queries the system struggled with — exactly what
        the basis needs to expand to cover.
        """
        try:
            entry = {
                'timestamp': time.time(),
                'query': query,
                'basis_words': [(w, float(e)) for w, e, _ in basis_words],
                'boundary_words': [(w, float(e)) for w, e, _ in boundary_words],
                'unknown_words': [(w, float(e)) for w, e, _ in unknown_words],
                'composed': success,
            }
            os.makedirs(os.path.dirname(self._boundary_log_path), exist_ok=True)
            with open(self._boundary_log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass  # Logging should never break the model


class ComputationDelegate:
    """Delegate computation to Python when the query maps to executable process.

    The model doesn't simulate computation — it translates to Python,
    executes deterministically, and translates back.
    """

    # Patterns that suggest executable computation
    COMPUTE_PATTERNS = [
        # Math expressions
        (re.compile(r'(?:what is|calculate|compute|evaluate|solve)\s+(.+)', re.I),
         'math'),
        # Sorting/list operations
        (re.compile(r'sort\s+(.+?)(?:\s+in\s+(\w+)\s+order)?', re.I),
         'sort'),
        # Counting
        (re.compile(r'how many\s+(\w+)\s+(?:are |)in\s+(.+)', re.I),
         'count'),
        # String operations
        (re.compile(r'reverse\s+(?:the\s+)?(?:string|word|text)\s+["\']?(.+?)["\']?$', re.I),
         'reverse'),
        # Unit conversion
        (re.compile(r'convert\s+(.+?)\s+to\s+(.+)', re.I),
         'convert'),
    ]

    # Safe builtins for sandboxed execution
    SAFE_BUILTINS = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sum': sum, 'len': len, 'sorted': sorted, 'reversed': reversed,
        'list': list, 'int': int, 'float': float, 'str': str,
        'range': range, 'enumerate': enumerate, 'zip': zip,
        'all': all, 'any': any, 'map': map, 'filter': filter,
        'True': True, 'False': False, 'None': None,
        'pow': pow, 'divmod': divmod, 'bool': bool,
    }

    def try_compute(self, query, trace=None):
        """Attempt to delegate to Python computation. Returns (result, used)."""
        if trace is None:
            trace = []

        # Try to extract a computable expression
        code, kind = self._extract_computation(query)
        if code is None:
            return None, False

        trace.append(f"[Compute] Extracted {kind}: {code[:80]}")

        # Execute in sandbox
        result = self._safe_execute(code, trace)
        if result is not None:
            trace.append(f"[Compute] Result: {result}")
            return str(result), True

        return None, False

    def _extract_computation(self, query):
        """Extract executable Python from natural language."""
        lower = query.lower().strip().rstrip('?.')

        # Fibonacci (check before math to prevent "10th" being parsed as "10")
        fib_match = re.search(r'(\d+)(?:th|st|nd|rd)\s+fibonacci', lower)
        if fib_match:
            n = int(fib_match.group(1))
            if n <= 100:
                return (f"(lambda n: (lambda f: f(f, n))"
                        f"(lambda f, n: n if n <= 1 else f(f, n-1) + f(f, n-2)))({n})"),\
                       'fibonacci'

        # Factorial (check before math)
        fact_match = re.search(r'factorial\s+(?:of\s+)?(\d+)', lower)
        if fact_match:
            n = int(fact_match.group(1))
            if n <= 20:
                return (f"(lambda n: (lambda f: f(f, n))"
                        f"(lambda f, n: 1 if n <= 1 else n * f(f, n-1)))({n})"),\
                       'factorial'

        # Prime check (before math)
        prime_match = re.search(r'is\s+(\d+)\s+(?:a\s+)?prime', lower)
        if prime_match:
            n = int(prime_match.group(1))
            return (f"all({n} % i != 0 for i in range(2, int({n}**0.5)+1)) "
                    f"and {n} > 1"), 'prime'

        # Direct math expression
        math_match = re.search(
            r'(?:what is|calculate|compute|evaluate)\s+(.+)', lower)
        if math_match:
            expr = math_match.group(1).strip().rstrip('?.')
            # Clean up natural language math
            expr = expr.replace('times', '*').replace('multiplied by', '*')
            expr = expr.replace('divided by', '/').replace('plus', '+')
            expr = expr.replace('minus', '-').replace('to the power of', '**')
            expr = expr.replace('squared', '**2').replace('cubed', '**3')
            expr = expr.replace('modulo', '%').replace('mod', '%')
            # Remove words, keep math
            cleaned = re.sub(r'[a-zA-Z]+', '', expr).strip()
            if cleaned and re.match(r'^[\d\s\+\-\*\/\%\.\(\)\,\*]+$', cleaned):
                return cleaned, 'math'
            # If the expression itself looks like math after cleanup
            if re.match(r'^[\d\s\+\-\*\/\%\.\(\)\,\*]+$', expr):
                return expr, 'math'

        # Sort request
        sort_match = re.search(
            r'sort\s+(?:these\s+)?(?:numbers?\s+)?(.+?)(?:\s+in\s+(\w+)\s+order)?$',
            lower)
        if sort_match:
            items_str = sort_match.group(1)
            order = sort_match.group(2) or ''
            nums = re.findall(r'-?\d+\.?\d*', items_str)
            if nums:
                nums_list = [float(n) if '.' in n else int(n) for n in nums]
                # Check for descending indicators anywhere in query
                reverse = (order.startswith('desc') or
                           'descend' in lower or 'largest' in lower or
                           'biggest' in lower or 'highest' in lower)
                return f"sorted({nums_list}, reverse={reverse})", 'sort'

        # Reverse string — capture text after "reverse (the string/word/text)"
        rev_match = re.search(
            r'reverse\s+(?:the\s+)?(?:string|word|text|sentence)?\s*["\']?(.+?)["\']?\s*$',
            lower)
        if rev_match and 'sort' not in lower:
            text = rev_match.group(1).strip().strip('"\'')
            if len(text) < 200 and not any(c in text for c in '(){}[]'):
                return f"'{text}'[::-1]", 'reverse'

        # Count occurrences
        count_match = re.search(
            r'how many\s+(\w)\s+(?:are\s+)?in\s+["\']?(\w+)["\']?', lower)
        if count_match:
            char = count_match.group(1)
            word = count_match.group(2)
            return f"'{word}'.count('{char}')", 'count'

        return None, None

    def _safe_execute(self, code, trace):
        """Execute code in a restricted sandbox. No imports, no I/O."""
        try:
            # Reject anything dangerous
            dangerous = ['import', 'exec', 'eval', 'open', 'file', '__',
                         'system', 'subprocess', 'os.', 'shutil', 'glob',
                         'input', 'breakpoint', 'compile']
            code_lower = code.lower()
            for d in dangerous:
                if d in code_lower:
                    trace.append(f"[Compute] Rejected: contains '{d}'")
                    return None

            result = eval(code, {"__builtins__": self.SAFE_BUILTINS}, {})
            return result
        except Exception as e:
            trace.append(f"[Compute] Execution error: {e}")
            return None


# Stop words for concept extraction
_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
    'can', 'could', 'may', 'might', 'shall', 'should', 'must',
    'not', 'no', 'nor', 'but', 'or', 'and', 'if', 'then', 'than',
    'too', 'also', 'very', 'just', 'only', 'even', 'still',
    'there', 'here', 'with', 'from', 'for', 'about', 'into',
    'what', 'how', 'who', 'why', 'where', 'when', 'which',
    'that', 'this', 'these', 'those', 'some', 'any', 'all',
    'more', 'most', 'less', 'much', 'many', 'few', 'other',
    'your', 'their', 'our', 'its', 'his', 'her', 'my',
    'tell', 'explain', 'describe', 'give', 'make', 'does',
})
