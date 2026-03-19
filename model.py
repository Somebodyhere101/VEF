"""
VEF — Language Model from Frozen Statistics
=============================================
Zero gradient descent. Fully interpretable.

Architecture:
    Embeddings (PPMI+SVD) → Retrieval → Introspection → Response
    With specialized paths for: arithmetic, composition, decomposition

Every answer traces to a specific corpus entry.
Every decision is logged in the reasoning trace.
"""
import os
import re
import pickle
import numpy as np

from tokenizers import Tokenizer

from core import Embeddings, Corpus, Attention, Relations, Refinement, Awareness, FusedOperators, DeepFusion, WordDecoder
from core.config import DEFAULT as CFG
from reasoning import Retrieval, Introspection, Arithmetic, Composition, Decomposition, Understanding, Circuits
from reasoning.boundary import BoundaryComposer, ComputationDelegate
from reasoning.instructions import InstructionFollower

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class VEF:
    """Language model from frozen statistics."""

    # Pre-compiled regex for identity claim detection (used by _align_identity)
    _identity_pattern = re.compile(
        r'\b(I am|my name is|I was|trained by|created by|developed by|made by|built by)\b',
        re.IGNORECASE)

    def __init__(self, data_dir=None, quiet=False):
        import time
        t0 = time.perf_counter()
        self._quiet = quiet

        data = data_dir or DATA_DIR

        # Core
        self.tokenizer = Tokenizer.from_file(os.path.join(data, 'tokenizer.json'))
        try:
            from tokenizers.decoders import ByteLevel
            if self.tokenizer.decoder is None:
                self.tokenizer.decoder = ByteLevel()
        except ImportError:
            pass

        # Suppress loading output in quiet mode
        import io, contextlib
        out = io.StringIO() if quiet else None
        with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
            self.embeddings = Embeddings(data)
            self.corpus = Corpus(data)
            self.attention = Attention(self.embeddings)
            self.relations = Relations(self.corpus.entries,
                                        max_entries=min(len(self.corpus.entries), 30000),
                                        cache_dir=data)
            self.relations._corpus_entries = self.corpus.entries
            self.relations.learn_antonym_axis(self.embeddings, self.tokenizer)
            self.relations._corpus_entries = None  # release large corpus ref
            self.awareness = Awareness(self.embeddings, self.tokenizer)
            self.refinement = Refinement(self.embeddings, self.corpus, self.tokenizer)

            def_path = os.path.join(data, 'definitions.pkl')
            if os.path.exists(def_path):
                with open(def_path, 'rb') as f:
                    self.definitions = pickle.load(f)
            else:
                self.definitions = {}

            self.retrieval = Retrieval(self.embeddings, self.corpus, self.tokenizer, self.attention, data_dir=data)
            self.introspection = Introspection(self.embeddings, self.corpus, self.tokenizer)
            self.arithmetic = Arithmetic(self.corpus.entries, cache_dir=data)
            self.composition = Composition(self.embeddings, self.corpus, self.tokenizer)
            self.decomposition = Decomposition()
            self.understanding = Understanding(self.corpus, self.embeddings, self.tokenizer, self.relations, self.retrieval)
            self.circuits = Circuits(self.embeddings, self.corpus, self.tokenizer, self.retrieval)
            self.boundary = BoundaryComposer(self.embeddings, self.corpus, self.tokenizer, self.awareness, self.retrieval)
            self.compute = ComputationDelegate()
            self.decoder = WordDecoder(self.embeddings, self.corpus, self.tokenizer, data_dir=data)
            self.fused = FusedOperators(self.embeddings, self.tokenizer, data_dir=data, corpus=self.corpus)
            self.fused.decoder = self.decoder  # Inject decoder
            self.deep_fusion = DeepFusion(self.embeddings, self.corpus, self.tokenizer, self.fused, self.awareness)
            self.deep_fusion.decoder = self.decoder  # Inject decoder
            self.instructor = InstructionFollower(
                self.embeddings, self.corpus, self.tokenizer, self.awareness,
                self.retrieval, self.deep_fusion, self.fused, self.decoder)

        self._system_prompt = None
        self._show_reasoning = False

        self.load_time = time.perf_counter() - t0

    def set_system(self, prompt, show_reasoning=False):
        """Set the system prompt — blended into every retrieval query.
        The embedding blend steers ALL responses toward the instructed behavior.
        """
        self._system_prompt = prompt
        self._show_reasoning = show_reasoning
        self.retrieval.set_system(prompt)

    def reason(self, query):
        """Main entry point. All circuits compete — no hardcoded priority.

        Every circuit that CAN answer produces a candidate.
        Each candidate is scored by confidence (embedding alignment to query).
        The highest-confidence answer wins.
        """
        q = query.strip()
        if not q:
            return ""

        trace = []
        candidates = []  # (confidence, answer, source)

        # Decompose compound queries first — each part gets full treatment
        parts = self.decomposition.split(q)
        if len(parts) > 1:
            trace.append(f"[Decomposition] Split into {len(parts)} parts")
            part_answers = []
            for part in parts:
                part_answer = self._compete(part, [])
                if part_answer and "don't have knowledge" not in part_answer:
                    part_answers.append(part_answer)
            if part_answers:
                return self._format(' '.join(part_answers), trace)

        # Single query — all circuits compete
        answer = self._compete(q, trace)
        return self._format(answer or "", trace)

    def _compete(self, query, trace):
        """All circuits produce candidates. Best confidence wins.

        Each circuit runs exactly once here; results are passed to _answer
        so it doesn't re-run them.
        """
        candidates = []
        q_emb = self.embeddings.embed(query, self.tokenizer)

        # Every circuit gets a chance to answer — run each once

        # Instruction following — parse and execute structured instructions
        instr_result, instr_used = self.instructor.follow(query, trace)
        if instr_used and instr_result:
            candidates.append((self._score(instr_result, q_emb) + 1.6,
                               instr_result, "Instruction"))

        # Deep fusion — multi-step reasoning inside the basis
        deep_result, deep_used = self.deep_fusion.reason(query, trace)
        if deep_used and deep_result:
            candidates.append((self._score(deep_result, q_emb) + 1.8,
                               deep_result, "Deep Fusion"))

        # Fused computation — single-step operators inside the embedding space
        fused_result, fused_used = self.fused.compute(query, trace)
        if fused_used and fused_result:
            candidates.append((self._score(fused_result, q_emb) + 1.5,
                               fused_result, "Fused Operator"))

        # Computation delegation — deterministic Python execution (fallback)
        # Gets a large bonus because computed answers are PROVABLY correct
        computed_delegate, delegate_used = self.compute.try_compute(query, trace)
        if delegate_used and computed_delegate:
            candidates.append((self._score(computed_delegate, q_emb) + 2.0,
                               computed_delegate, "Computation Delegate"))

        # Boundary-driven composition — novel answers from partial knowledge
        # Gets a bonus for covering concepts that no single retrieval result does
        boundary_resp, boundary_used = self.boundary.try_compose(query, trace)
        if boundary_used and boundary_resp:
            base_score = self._score(boundary_resp, q_emb)
            # Coverage bonus: count query content words in the composed response
            content = self.corpus.content_words(query)
            if content:
                resp_lower = boundary_resp.lower()
                coverage = sum(1 for w in content if w in resp_lower) / len(content)
                base_score = base_score * (1.0 + 0.8 * coverage)
            candidates.append((base_score, boundary_resp, "Boundary Composition"))

        # Word problems
        wp = self.understanding.solve_word_problem(query)
        if wp:
            candidates.append((self._score(wp, q_emb), wp, "Word Problem"))

        # Logic (antonyms, sequences, etc.)
        logic = self.understanding.answer_logic(query)
        if logic:
            candidates.append((self._score(logic, q_emb), logic, "Logic"))

        # Arithmetic / letter counting — call once, reuse below
        computed = self._try_computation(query, trace)
        if computed:
            candidates.append((self._score(computed, q_emb), computed, "Computation"))

        # Definitions — use extracted helper, call once
        definition = self._lookup_definition(query)
        if definition:
            candidates.append((self._score(definition, q_emb), definition, "Definition"))

        # Composition (joke, poem, etc.)
        composed = self.composition.try_compose(query)
        if composed:
            candidates.append((self._score(composed, q_emb), composed, "Composition"))

        # Retrieval (the general fallback) — pass pre-computed results
        # so _answer doesn't re-run logic, word problems, definitions, or computation
        pre_computed = {
            'logic': logic,
            'word_problem': wp,
            'definition': definition,
            'computed': computed,
        }
        retrieval_answer = self._answer(query, trace, pre_computed=pre_computed)
        if retrieval_answer:
            candidates.append((self._score(retrieval_answer, q_emb),
                              retrieval_answer, "Retrieval"))

        if not candidates:
            return "I don't understand the query. Could you rephrase?"

        # Best confidence wins
        candidates.sort(key=lambda x: -x[0])
        best_conf, best_answer, best_source = candidates[0]
        trace.append(f"[{best_source}] Won with confidence {best_conf:.3f}")
        return best_answer

    def _lookup_definition(self, query):
        """Extract subject from a definition query and look up in definitions dict.

        Returns the definition string if found, or None.
        """
        if not self.definitions:
            return None
        lower = query.lower().rstrip('?. ')
        def_match = (
            re.match(r"what(?:'s| is| are) (?:a |an |the )?(.+)", lower) or
            re.match(r"(?:explain|describe|tell me about) (?:a |an |the )?(?:what )?(.+)", lower)
        )
        if not def_match:
            return None
        subject = def_match.group(1).strip()
        if (re.search(r'\d|[+\-*/]', subject)
            or re.search(r'\b(your|my|you|me)\b', subject)
            or len(subject) < 3):
            return None
        if subject in self.definitions:
            return self.definitions[subject]
        for word in subject.split():
            if len(word) >= 4 and word in self.definitions:
                return self.definitions[word]
        return None

    def _score(self, answer, q_emb):
        """Score a candidate answer by embedding alignment to the query."""
        if q_emb is None or not answer:
            return 0.0
        a_emb = self.embeddings.embed(answer[:200], self.tokenizer)
        if a_emb is None:
            return 0.0
        return float(np.dot(q_emb, a_emb))

    def _answer(self, query, trace, pre_computed=None):
        """Answer a single query through understanding + retrieval + introspection.

        When called from _compete, pre_computed contains already-evaluated circuit
        results to avoid duplicate work.
        """
        pre = pre_computed or {}

        # Detect format constraints from the query
        constraints = self.understanding.detect_constraints(query)
        if constraints:
            trace.append(f"[Constraints] {constraints}")

        # Try logic/reasoning first (antonyms, sequences, syllogisms)
        # Reuse pre-computed result if available
        logic = pre.get('logic') if pre.get('logic') is not None else self.understanding.answer_logic(query)
        if logic:
            trace.append(f"[Logic] Answered from learned patterns")
            return self.understanding.apply_constraints(logic, constraints, query)

        # Try word problem solving — reuse if available
        word_problem = pre.get('word_problem') if pre.get('word_problem') is not None else self.understanding.solve_word_problem(query)
        if word_problem:
            trace.append(f"[Word Problem] Parsed and computed")
            return word_problem

        # CATEGORY BASIS: check definitions — reuse extracted helper
        definition = pre.get('definition') if pre.get('definition') is not None else self._lookup_definition(query)
        if definition:
            trace.append(f"[Category] Found definition")
            return definition

        # Try computation BEFORE awareness — reuse if available
        computed = pre.get('computed') if pre.get('computed') is not None else self._try_computation(query, trace)
        if computed:
            return computed

        # Extract concept words
        concept_words = list(self.corpus.content_words(query))
        all_words = re.findall(r'[a-z]+', query.lower())
        unknown = [w for w in all_words if len(w) >= 3 and w not in self.corpus.word_index]
        concept_words = list(set(concept_words + unknown))
        if not concept_words:
            concept_words = [w for w in all_words if len(w) >= 3]
        trace.append(f"[Analysis] Concepts: {concept_words}")

        # AWARENESS: measure UNKNOWN concept words against basis/anti-basis
        # If the key concepts fall in the anti-basis, the model has no knowledge
        # about them — try spell correction or refuse honestly
        key_concepts = unknown if unknown else [w for w in concept_words
                                                 if w not in {'what','how','who','why','where',
                                                              'when','which','does','the','are'}]
        # If NO meaningful concepts found, or query looks malformed
        # (only structural words + symbols), don't hallucinate
        if not key_concepts:
            # Check: is there actual content or just noise/symbols?
            raw = re.sub(r'[^a-zA-Z\s]', '', query).strip()
            meaningful_words = [w for w in raw.lower().split() if len(w) >= 3
                                and w not in {'what','how','who','why','the','are','does','can'}]
            if not meaningful_words:
                trace.append("[Awareness] No meaningful content in query")
                return "I don't understand the query. Could you rephrase?"

        if key_concepts:
            noise_count = 0
            for cw in key_concepts:
                cat, energy, conc, detail = self.awareness.measure(cw)
                if cat == 'noise':
                    noise_count += 1
            if noise_count > 0 and noise_count >= len(key_concepts) * 0.5:
                trace.append(f"[Awareness] All concepts in anti-basis — attempting correction")
                # Try spell correction before refusing
                corrections = self.introspection.try_spell_correction(query, concept_words)
                if corrections:
                    corrected = query.lower()
                    for word, (match, dist) in corrections.items():
                        trace.append(f"[Thinking] \"{word}\" → \"{match}\" (edit distance {dist})")
                        corrected = corrected.replace(word, match)
                    corrected_answer = self._answer_corrected(corrected, trace)
                    if corrected_answer:
                        return corrected_answer
                # No correction possible — the model genuinely doesn't know
                topic = ' '.join(concept_words)
                trace.append(f"[Awareness] Anti-basis confirms: no knowledge of \"{topic}\"")
                return f"I don't have knowledge about \"{topic}\"."

        # Multi-hop retrieval: separates intent from concept, composes results
        best_resp = self.circuits.multi_hop(query, trace)
        if not best_resp:
            trace.append("[Retrieval] No results")
            return computed  # reuse already-computed result (may be None)

        # Also get the raw score for confidence measurement
        results = self.retrieval.search(query, top_k=5)
        best_score = results[0][0] if results else 0.0
        trace.append(f"[Retrieval] Best: \"{best_resp[:60]}...\" (score={best_score:.3f})")

        # Check if it's a math/letter question BEFORE accepting retrieval
        # Reuse the computation result from earlier
        if computed:
            return computed

        # Check concept coverage
        if concept_words:
            coverage = sum(1 for w in concept_words if w in best_resp.lower())
            if coverage > 0 and len(best_resp) > 30:
                trace.append(f"[Result] Coverage: {coverage}/{len(concept_words)}")
                result = self._clean(best_resp[:500])
                if constraints:
                    result = self.understanding.apply_constraints(result, constraints, query)
                return result

        # Introspection: measure confidence
        confidence, details = self.introspection.measure_confidence(
            query, best_resp, best_score)

        # If ALL concept words are unknown, heavily reduce confidence
        # (the model doesn't recognize what it's being asked about)
        if unknown and len(unknown) >= len([w for w in concept_words if w not in unknown]):
            confidence *= 0.2
            details += f" [unknown concepts: {unknown}]"

        trace.append(f"[Introspection] {details}, confidence={confidence:.3f}")

        if confidence > CFG.CONFIDENCE_ACCEPT:
            return self._clean(best_resp[:500])

        # Low confidence — try category lookup
        cat_result = self.circuits.category_lookup(query)
        if cat_result:
            trace.append("[Circuit] Category lookup")
            return cat_result

        # Unknown concepts? Try spell correction FIRST — correct perception before reasoning.
        # The brain fixes "did I hear that right?" before "what does it mean?"
        all_words = [w for w in re.findall(r'[a-z]+', query.lower()) if len(w) >= 3]
        if unknown:
            corrections = self.introspection.try_spell_correction(query, all_words)
            if corrections:
                corrected = query.lower()
                for word, (match, dist) in corrections.items():
                    trace.append(f"[Thinking] \"{word}\" → \"{match}\" (edit distance {dist})")
                    corrected = corrected.replace(word, match)

                def_match = re.match(r"what(?:'s| is| are) (?:a |an |the )?(.+)",
                                     corrected.rstrip('?. '))
                if def_match and self.definitions:
                    subject = def_match.group(1).strip()
                    if subject in self.definitions:
                        trace.append(f"[Correction] Found definition for '{subject}'")
                        return self.definitions[subject]

                corrected_answer = self._answer_corrected(corrected, trace)
                if corrected_answer:
                    return corrected_answer

        # Low confidence, no corrections — iterative refinement
        thought_result, thought_trace, n_steps = self.refinement.refine(query)
        if thought_result and len(thought_result) > 20:
            trace.append(f"[Refinement] Converged in {n_steps} steps")
            for t in thought_trace:
                trace.append(f"  {t}")
            t_emb = self.embeddings.embed(thought_result[:200], self.tokenizer)
            if t_emb is not None:
                q_emb_check = self.embeddings.embed(query, self.tokenizer)
                if q_emb_check is not None:
                    thought_conf = float(np.dot(q_emb_check, t_emb))
                    best_emb_check = self.embeddings.embed(best_resp[:200], self.tokenizer)
                    best_conf = float(np.dot(q_emb_check, best_emb_check)) if best_emb_check is not None else 0
                    if thought_conf > best_conf:
                        return self._clean(thought_result[:500])

        # Last resort: spell correction for known words too
        if not unknown:
            corrections = self.introspection.try_spell_correction(query, all_words)
            if corrections:
                corrected = query.lower()
                for word, (match, dist) in corrections.items():
                    trace.append(f"[Thinking] \"{word}\" → \"{match}\" (edit distance {dist})")
                    corrected = corrected.replace(word, match)
                corrected_answer = self._answer_corrected(corrected, trace)
                if corrected_answer:
                    return corrected_answer

        # Check partial knowledge
        known = self.introspection.find_partial_knowledge(
            concept_words, self.retrieval)
        if known:
            topic = ' '.join(concept_words)
            parts = [f"I'm not confident about \"{topic}\". Here's what I know:"]
            for word, info in known:
                parts.append(f"  {self._clean(info)}")
            return '\n'.join(parts)

        return f"I don't have knowledge about \"{' '.join(concept_words)}\"."

    def _answer_corrected(self, corrected_query, trace):
        """Re-run the answer pipeline on a spell-corrected query."""
        trace.append(f"[Correction] Retrying with: \"{corrected_query}\"")

        # Try definitions
        lower = corrected_query.rstrip('?. ')
        def_match = re.match(r"what(?:'s| is| are) (?:a |an |the )?(.+)", lower)
        if def_match and self.definitions:
            subject = def_match.group(1).strip()
            if subject in self.definitions:
                return self.definitions[subject]

        # Try logic
        logic = self.understanding.answer_logic(corrected_query)
        if logic:
            return logic

        # Try retrieval
        results = self.retrieval.search(corrected_query, top_k=5)
        for score, resp in results:
            if len(resp) > 20:
                return self._clean(resp[:500])

        return None

    def _try_computation(self, query, trace):
        """Try arithmetic or letter counting."""
        answer, detail = self.arithmetic.solve(query)
        if answer:
            trace.append(f"[Arithmetic] {detail}")
            if re.search(r'\bwhy\b|\bexplain\b', query.lower()) and detail:
                return f"{answer}\n\n{detail}"
            return answer

        # Letter introspection
        lower = query.lower()
        letter_match = (re.search(r"\b([a-zA-Z])'s\b", query) or
                        re.search(r'\bletter[s]?\s+([a-zA-Z])\b', lower) or
                        re.search(r'\b([a-zA-Z])s?\b(?=\s+in\b)', query))
        if letter_match and 'how many' in lower:
            letter = letter_match.group(1).lower()
            tokens = lower.split()
            target = None
            if 'in' in tokens:
                after = tokens[tokens.index('in') + 1:]
                for w in reversed(after):
                    clean = re.sub(r'[^a-z]', '', w)
                    if len(clean) >= 2:
                        target = clean
                        break
            if target:
                count = target.count(letter)
                spelled = ', '.join(target)
                trace.append(f"[Introspection] Letter counting: {letter} in {target}")
                return (f'The word "{target}" is spelled: {spelled}. '
                        f'Counting "{letter}": there {"are" if count != 1 else "is"} '
                        f'{count} "{letter.upper()}" in "{target}".')

        return None

    def _format(self, answer, trace):
        if not answer:
            return ""
        # Self-edit: align identity claims with the system prompt
        answer = self._align_identity(answer)
        if self._show_reasoning and trace:
            reasoning = '\n'.join(f'  {t}' for t in trace)
            return f"{answer}\n\n[Reasoning]\n{reasoning}"
        return answer

    def _align_identity(self, text):
        """Replace first-person identity claims in retrieved text with the system prompt.

        Only triggers on direct self-identification ("I am X", "my name is X",
        "trained by X") — not general uses of "I am" in other contexts.
        """
        if not self._system_prompt:
            return text
        if not self._identity_pattern.search(text):
            return text
        # Only replace if the identity claim is in the first sentence
        # (where retrieved responses typically state identity)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return text
        if self._identity_pattern.search(sentences[0]):
            sentences[0] = self._system_prompt
            return ' '.join(sentences)
        return text

    @staticmethod
    def _clean(text):
        # Remove only 3+ consecutive duplicate words; keep legitimate pairs
        # like "that that" or "had had"
        words = text.split()
        if not words:
            return text
        cleaned = [words[0]]
        for i in range(1, len(words)):
            # Count how many times this word has repeated consecutively
            run_length = 1
            j = i
            while j >= 1 and words[j].lower() == words[j-1].lower():
                run_length += 1
                j -= 1
            # Only skip if this would be the 3rd+ consecutive duplicate
            if run_length < 3:
                cleaned.append(words[i])
        text = ' '.join(cleaned)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        last_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_end > len(text) * 0.4:
            text = text[:last_end + 1]
        return re.sub(r'\s+', ' ', text).strip()
