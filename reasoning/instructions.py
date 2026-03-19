"""
Instruction Following — parse and execute novel instructions.

Any instruction can be decomposed into:
  1. WHAT (the subject/target concepts)
  2. HOW (the operation to apply)
  3. CONSTRAINTS (format, count, style)

The instruction parser extracts these, maps HOW to operator chains,
applies them to WHAT, and formats per CONSTRAINTS.

"List 3 animals bigger than a cat"
→ WHAT: animals  HOW: filter(bigger_than, cat)  CONSTRAINTS: count=3, format=list

"Explain gravity in simple terms"
→ WHAT: gravity  HOW: simplify  CONSTRAINTS: style=simple

"Compare Python and JavaScript"
→ WHAT: [python, javascript]  HOW: intersection + difference  CONSTRAINTS: format=comparison
"""
import re
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InstructionFollower:
    """Parse and execute novel instructions via operator chains."""

    def __init__(self, embeddings, corpus, tokenizer, awareness,
                 retrieval, deep_fusion, fused_ops, decoder=None):
        self.embeddings = embeddings
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.awareness = awareness
        self.retrieval = retrieval
        self.deep_fusion = deep_fusion
        self.fused_ops = fused_ops
        self.decoder = decoder

        # Build instruction operator vocabulary
        # Each instruction verb maps to a reasoning strategy
        self.instruction_ops = {
            # Information retrieval
            'explain': self._op_explain,
            'describe': self._op_explain,
            'define': self._op_explain,
            # Comparison
            'compare': self._op_compare,
            'contrast': self._op_compare,
            'difference': self._op_compare,
            # Enumeration
            'list': self._op_list,
            'name': self._op_list,
            'give': self._op_list,
            'examples': self._op_list,
            # Transformation
            'simplify': self._op_simplify,
            'summarize': self._op_simplify,
            'translate': self._op_transform,
            'rewrite': self._op_transform,
            'convert': self._op_transform,
            # Reasoning
            'why': self._op_reason,
            'how': self._op_reason,
            'cause': self._op_reason,
            # Creative
            'create': self._op_create,
            'write': self._op_create,
            'generate': self._op_create,
            'invent': self._op_create,
            # Analysis
            'analyze': self._op_analyze,
            'evaluate': self._op_analyze,
            'assess': self._op_analyze,
            # Classification
            'classify': self._op_classify,
            'categorize': self._op_classify,
            'identify': self._op_classify,
        }

    def follow(self, query, trace=None):
        """Parse and execute an instruction. Returns (response, used)."""
        if trace is None:
            trace = []

        # Step 1: Parse the instruction
        parsed = self._parse(query)
        if parsed is None:
            return None, False

        op_name, subjects, constraints = parsed
        trace.append(f"[Instruction] Parsed: op={op_name}, "
                     f"subjects={subjects[:3]}, constraints={constraints}")

        # Step 2: Find the operation handler
        handler = self.instruction_ops.get(op_name)
        if handler is None:
            # Try fuzzy match via embedding similarity
            handler = self._fuzzy_match_op(op_name)

        if handler is None:
            return None, False

        # Step 3: Execute
        result = handler(subjects, constraints, query, trace)
        if result is None:
            return None, False

        # Step 3.5: Reject code responses for non-code queries
        if result and self._is_code_response(result):
            if not any(w in query.lower() for w in ['code', 'program', 'function',
                                                      'implement', 'algorithm']):
                trace.append(f"[Instruction] Rejected code response")
                return None, False

        # Step 4: Apply constraints
        result = self._apply_constraints(result, constraints, trace)

        trace.append(f"[Instruction] Executed: {op_name} → {len(result)} chars")
        return result, True

    def _parse(self, query):
        """Parse an instruction into (operation, subjects, constraints)."""
        lower = query.lower().strip().rstrip('?.')

        # Extract constraints first
        constraints = {}

        # Count constraint: "list 3" / "name 5" / "give me 10"
        count_match = re.search(r'(?:list|name|give|provide|show)\s+(\d+)', lower)
        if count_match:
            constraints['count'] = int(count_match.group(1))

        # Style constraint: "in simple terms" / "briefly" / "in detail"
        if 'simple' in lower or 'simply' in lower or 'easy' in lower:
            constraints['style'] = 'simple'
        elif 'brief' in lower or 'short' in lower or 'concise' in lower:
            constraints['style'] = 'brief'
        elif 'detail' in lower or 'thorough' in lower:
            constraints['style'] = 'detailed'

        # Format constraint: "as a list" / "in a table" / "step by step"
        if 'step by step' in lower or 'steps' in lower:
            constraints['format'] = 'steps'
        elif 'list' in lower and 'count' not in constraints:
            constraints['format'] = 'list'

        # Detect the operation verb
        op_name = None
        subjects = []

        # Pattern: "verb + subject" or "verb + subject + modifier"
        for verb in sorted(self.instruction_ops.keys(), key=len, reverse=True):
            if verb in lower:
                op_name = verb
                # Extract what comes after the verb
                verb_pos = lower.index(verb)
                after = lower[verb_pos + len(verb):].strip()
                # Remove common filler
                after = re.sub(
                    r'^(?:me |us |the |a |an |about |of |for |between |to )+',
                    '', after).strip()
                if after:
                    # Split on "and" or "," for multiple subjects
                    parts = re.split(r'\s+and\s+|\s*,\s*', after)
                    subjects = [p.strip().rstrip('?.!') for p in parts
                                if len(p.strip()) >= 2]
                break

        if op_name is None:
            return None

        # Clean subjects: remove constraint text
        clean_subjects = []
        constraint_words = {'simple', 'simply', 'brief', 'briefly', 'detail',
                            'detailed', 'step', 'steps', 'list', 'concise',
                            'easy', 'terms', 'short'}
        for s in subjects:
            cleaned = ' '.join(w for w in s.split()
                               if w.lower() not in constraint_words)
            if len(cleaned) >= 2:
                clean_subjects.append(cleaned)

        if not clean_subjects:
            # Try to extract subject from the full query
            content_words = [w for w in re.findall(r'[a-z]{3,}', lower)
                             if w not in _INSTRUCTION_WORDS and w != op_name]
            if content_words:
                clean_subjects = [' '.join(content_words)]

        return op_name, clean_subjects, constraints

    def _fuzzy_match_op(self, op_name):
        """Find closest operation via embedding similarity."""
        op_emb = self.embeddings.embed(op_name, self.tokenizer)
        if op_emb is None:
            return None

        best_handler = None
        best_sim = -1

        for known_op, handler in self.instruction_ops.items():
            known_emb = self.embeddings.embed(known_op, self.tokenizer)
            if known_emb is not None:
                sim = float(np.dot(op_emb, known_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_handler = handler

        if best_sim > 0.6:
            return best_handler
        return None

    # ================================================================
    # Operation Handlers
    # ================================================================

    def _op_explain(self, subjects, constraints, query, trace):
        """Explain/describe/define a concept."""
        if not subjects:
            return None

        subject = ' '.join(subjects)
        results = self.retrieval.search(subject, top_k=5)
        if not results:
            return None

        best = results[0][1]

        # If style=simple, extract just the first sentence
        if constraints.get('style') == 'simple':
            sentences = re.split(r'[.!?]+', best)
            simple = [s.strip() for s in sentences if len(s.strip()) > 10]
            if simple:
                best = simple[0] + '.'

        # If style=brief, truncate
        if constraints.get('style') == 'brief' and len(best) > 200:
            best = best[:200].rsplit('.', 1)[0] + '.'

        return best

    def _op_compare(self, subjects, constraints, query, trace):
        """Compare two or more concepts."""
        if len(subjects) < 2:
            # Try to split a single subject on "and"/"vs"/"or"
            if subjects:
                parts = re.split(r'\s+(?:and|vs|versus|or)\s+', subjects[0])
                if len(parts) >= 2:
                    subjects = [p.strip() for p in parts]

        if len(subjects) < 2:
            return None

        a, b = subjects[0], subjects[1]
        ea = self.embeddings.embed(a, self.tokenizer)
        eb = self.embeddings.embed(b, self.tokenizer)
        if ea is None or eb is None:
            return None

        # Similarity
        similarity = float(np.dot(
            ea / (np.linalg.norm(ea) + 1e-10),
            eb / (np.linalg.norm(eb) + 1e-10)))

        # What they share (intersection)
        shared_emb = (ea + eb) / 2
        shared_emb = shared_emb / (np.linalg.norm(shared_emb) + 1e-10)
        shared_words = self._decode(shared_emb, exclude={a, b} |
                                     set(a.split()) | set(b.split()))

        # What's different (each concept's unique direction)
        diff = ea - eb
        diff_n = diff / (np.linalg.norm(diff) + 1e-10)
        a_unique = self._decode(ea + diff_n * 0.3, exclude={a, b} |
                                 set(a.split()) | set(b.split()))
        b_unique = self._decode(eb - diff_n * 0.3, exclude={a, b} |
                                 set(a.split()) | set(b.split()))

        # Get context from corpus (use subject-specific search, not code-heavy)
        a_info = self._get_first_sentence(f"{a} programming language")
        b_info = self._get_first_sentence(f"{b} programming language")
        if not a_info:
            a_info = self._get_first_sentence(a)
        if not b_info:
            b_info = self._get_first_sentence(b)

        trace.append(f"[Instruction] Compare: similarity={similarity:.3f}, "
                     f"shared={shared_words[:3]}")

        # Build comparison — use the instruction result, not retrieval
        parts = []
        if a_info:
            parts.append(f"{a.capitalize()}: {a_info}")
        if b_info:
            parts.append(f"{b.capitalize()}: {b_info}")
        if shared_words:
            parts.append(f"Both relate to {', '.join(shared_words[:3])}.")
        if a_unique and b_unique:
            parts.append(f"{a.capitalize()} is more associated with "
                         f"{', '.join(a_unique[:2])}, "
                         f"while {b} is more associated with "
                         f"{', '.join(b_unique[:2])}.")
        parts.append(f"Overall similarity: {similarity:.0%}.")

        return ' '.join(parts)

    def _op_list(self, subjects, constraints, query, trace):
        """List examples or members of a category."""
        if not subjects:
            return None

        subject = ' '.join(subjects)
        count = constraints.get('count', 5)

        # Strategy: search corpus for entries that mention this category
        # and extract specific instances from those entries
        import re as _re

        # Search for "[instance] is a/an [subject]" patterns
        subject_words = set(subject.lower().split())
        members = set()

        # Search corpus entries near the category embedding
        emb = self.embeddings.embed(subject, self.tokenizer)
        if emb is None:
            return None

        q_t = torch.tensor(emb, dtype=torch.float32, device=DEVICE)
        scores = self.corpus.q_embeds @ q_t
        top_k = min(200, len(scores))
        _, top_idx = scores.topk(top_k)

        for idx in top_idx:
            idx_val = idx.item()
            if idx_val >= len(self.corpus.entries):
                continue
            entry = self.corpus.entries[idx_val].lower()

            # Pattern: "[X] is a/an [category_word]"
            for sw in subject_words:
                if len(sw) < 3:
                    continue
                pattern = rf'(\w+)\s+is\s+(?:a|an)\s+{_re.escape(sw)}'
                for m in _re.finditer(pattern, entry):
                    candidate = m.group(1)
                    if (len(candidate) >= 3 and candidate not in subject_words
                            and candidate not in _INSTRUCTION_WORDS
                            and candidate.isalpha()):
                        members.add(candidate)

            # Pattern: "[category]: [X], [Y], [Z]"
            for sw in subject_words:
                if sw in entry:
                    # Look for list-like patterns near the category word
                    list_match = _re.search(
                        rf'{_re.escape(sw)}[s]?\s*(?:include|such as|like|are|:)\s*(.+?)(?:\.|$)',
                        entry)
                    if list_match:
                        items = _re.findall(r'[a-z]{3,}', list_match.group(1))
                        for item in items[:5]:
                            if (item not in subject_words and
                                    item not in _INSTRUCTION_WORDS):
                                members.add(item)

            if len(members) >= count * 2:
                break

        if members:
            # Rank members by embedding similarity to the category
            scored = []
            for m in members:
                m_emb = self.embeddings.embed(m, self.tokenizer)
                if m_emb is not None:
                    sim = float(np.dot(
                        m_emb / (np.linalg.norm(m_emb) + 1e-10),
                        emb / (np.linalg.norm(emb) + 1e-10)))
                    if 0.2 < sim < 0.95:
                        scored.append((sim, m))

            scored.sort(key=lambda x: -x[0])
            result_members = [w for _, w in scored[:count]]

            if result_members:
                trace.append(f"[Instruction] List: {subject} → {result_members}")
                return ', '.join(w.capitalize() for w in result_members) + '.'

        # Fallback: nearest embedding neighbors (less reliable)
        candidates = self._decode(emb, exclude=subject_words, top_k=count)
        if candidates:
            trace.append(f"[Instruction] List (fallback): {subject} → {candidates}")
            return ', '.join(w.capitalize() for w in candidates) + '.'

        return None

    def _op_simplify(self, subjects, constraints, query, trace):
        """Simplify/summarize a concept."""
        constraints['style'] = 'simple'
        return self._op_explain(subjects, constraints, query, trace)

    def _op_transform(self, subjects, constraints, query, trace):
        """Transform text (translate, rewrite, convert)."""
        # This is limited without a generative model, but we can
        # use the embedding space to find related forms
        if not subjects:
            return None

        subject = ' '.join(subjects)

        # Check if this is a concept translation
        # "translate gravity to simple terms" → explain simply
        if 'simple' in query.lower():
            constraints['style'] = 'simple'
            return self._op_explain(subjects, constraints, query, trace)

        return None

    def _op_reason(self, subjects, constraints, query, trace):
        """Why/how reasoning via causal operator chains."""
        if not subjects:
            return None

        subject = ' '.join(subjects)

        # Try deep fusion's conditional reasoning
        if self.deep_fusion:
            result, used = self.deep_fusion.reason(query, trace)
            if used and result:
                return result

        # Fallback: retrieve
        results = self.retrieval.search(query, top_k=3)
        if results:
            return results[0][1]

        return None

    def _op_create(self, subjects, constraints, query, trace):
        """Creative generation via form×content composition."""
        if not subjects:
            return None

        subject = ' '.join(subjects)

        # Determine what kind of creation
        lower = query.lower()
        if 'analogy' in lower or 'metaphor' in lower:
            # Create an analogy: find what the subject is similar to
            # in an unexpected domain
            emb = self.embeddings.embed(subject, self.tokenizer)
            if emb is None:
                return None

            # Find distant but connected concepts
            candidates = self._decode(emb, exclude=set(subject.split()), top_k=20)
            # Skip the first few (too obvious) and pick from the middle
            if len(candidates) > 5:
                unexpected = candidates[5:10]
                return (f"{subject.capitalize()} is like {unexpected[0]} — "
                        f"both involve {candidates[0]} and {candidates[1]}.")

        # General creation: find relevant corpus entries and compose
        results = self.retrieval.search(subject, top_k=3)
        if results:
            return results[0][1]

        return None

    def _op_analyze(self, subjects, constraints, query, trace):
        """Analyze a concept by decomposing into components."""
        if not subjects:
            return None

        subject = ' '.join(subjects)
        emb = self.embeddings.embed(subject, self.tokenizer)
        if emb is None:
            return None

        # Awareness check: how well does the model know this?
        cat, energy, conc, detail = self.awareness.measure(subject)

        # Find component concepts
        components = self._decode(emb, exclude=set(subject.split()), top_k=8)

        # Get corpus info
        info = self._get_first_sentence(subject)

        parts = []
        if info:
            parts.append(info)
        if components:
            parts.append(f"Key aspects: {', '.join(components[:5])}.")
        parts.append(f"Model confidence: {cat} "
                     f"(energy={energy:.2f}, concentration={conc:.2f}).")

        trace.append(f"[Instruction] Analyze: {subject} → {cat}, "
                     f"components={components[:5]}")
        return ' '.join(parts)

    def _op_classify(self, subjects, constraints, query, trace):
        """Classify a concept using the category operator."""
        if not subjects:
            return None

        subject = ' '.join(subjects)

        # Use fused category operator if available
        if (self.fused_ops and 'categorize' in self.fused_ops.operators):
            emb = self.embeddings.embed(subject, self.tokenizer)
            if emb is not None:
                cat_op = self.fused_ops.operators['categorize']
                result_emb = emb + cat_op
                result_emb = result_emb / (np.linalg.norm(result_emb) + 1e-10)
                category = self._decode(result_emb,
                                        exclude=set(subject.split()),
                                        top_k=1)
                if category:
                    trace.append(f"[Instruction] Classify: {subject} → {category[0]}")
                    return f"{subject.capitalize()} is a type of {category[0]}."

        return None

    # ================================================================
    # Helpers
    # ================================================================

    def _decode(self, emb, exclude=None, top_k=5):
        """Decode embedding to words using best available decoder."""
        if exclude is None:
            exclude = set()

        full_exclude = exclude | _INSTRUCTION_WORDS

        if self.decoder:
            return self.decoder.decode_words(emb, top_k=top_k, exclude=full_exclude)

        # Fallback
        sims = self.embeddings.normed @ emb
        n = min(50, len(sims))
        top_indices = np.argpartition(-sims, n)[:n]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        results = []
        for idx in top_indices:
            decoded = self.tokenizer.decode([int(idx)]).strip().lower()
            decoded = decoded.replace('Ġ', '').replace('Ã', '').strip()
            if (len(decoded) >= 3 and decoded.isalpha() and
                    decoded not in full_exclude):
                results.append(decoded)
                if len(results) >= top_k:
                    break
        return results

    def _get_first_sentence(self, concept):
        """Get the first sentence from the best corpus match."""
        results = self.retrieval.search(concept, top_k=1)
        if results:
            resp = results[0][1]
            sentences = re.split(r'[.!?]+', resp)
            for s in sentences:
                s = s.strip()
                if len(s) > 15:
                    return s + '.'
        return None

    def _is_code_response(self, text):
        """Check if a response is code rather than natural language."""
        code_indicators = ['def ', 'return ', 'import ', 'class ', 'for i in',
                           'if (', '= []', '.append(', 'function ', 'var ',
                           'console.', '"""', "'''", '```']
        text_lower = text[:300].lower()
        return sum(1 for c in code_indicators if c.lower() in text_lower) >= 2

    def _apply_constraints(self, result, constraints, trace):
        """Apply formatting constraints to the result."""
        if not result:
            return result

        style = constraints.get('style')
        if style == 'simple' and len(result) > 200:
            # Truncate to first 1-2 sentences
            sentences = re.split(r'(?<=[.!?])\s+', result)
            result = ' '.join(sentences[:2])

        if style == 'brief' and len(result) > 150:
            result = result[:150].rsplit('.', 1)[0] + '.'

        fmt = constraints.get('format')
        if fmt == 'steps':
            # Convert to numbered steps
            sentences = [s.strip() for s in re.split(r'[.!?]+', result)
                         if len(s.strip()) > 10]
            if sentences:
                result = ' '.join(f"{i+1}. {s}." for i, s in enumerate(sentences[:5]))

        return result


_INSTRUCTION_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'and', 'or', 'but', 'not', 'no', 'to', 'of', 'in', 'for',
    'on', 'at', 'by', 'it', 'he', 'she', 'we', 'they', 'this',
    'that', 'with', 'from', 'as', 'do', 'has', 'have', 'had',
    'me', 'us', 'please', 'can', 'you', 'will', 'could', 'would',
    'what', 'how', 'why', 'who', 'where', 'when', 'which',
    'explain', 'describe', 'define', 'list', 'name', 'give',
    'compare', 'contrast', 'simplify', 'summarize', 'analyze',
    'create', 'write', 'generate', 'classify', 'categorize',
    'translate', 'rewrite', 'convert', 'identify', 'evaluate',
})
