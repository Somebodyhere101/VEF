"""
Deep Fusion — chained computation inside the embedding space.

Single operators (dog→animal) are shallow. Understanding requires:
  1. Decomposing a query into a computation graph
  2. Executing each node via operator transformations
  3. Feeding outputs to the next operator
  4. Decoding the final embedding as the answer

"If photosynthesis worked in reverse, what would it produce?"
→ embed(photosynthesis) → apply(reverse_process) → apply(extract_outputs) → decode

The embedding space becomes a virtual machine.
Each dimension is a register. Each operator is an instruction.
The query programs the machine. The answer is the final state.
"""
import re
import numpy as np
import torch
from collections import defaultdict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeepFusion:
    """Chain operators for multi-step reasoning inside the basis."""

    def __init__(self, embeddings, corpus, tokenizer, operators, awareness):
        self.embeddings = embeddings
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.operators = operators
        self.awareness = awareness
        self.dim = embeddings.dim

        # Build the subspace intersection engine
        self._build_concept_subspaces()

    def _build_concept_subspaces(self):
        """For each concept, identify which SVD dimensions it activates most.

        Two concepts "share understanding" when their active dimensions overlap.
        The intersection IS what they have in common — computed geometrically.
        """
        self._concept_signatures = {}  # word → top-k dimension indices

    def reason(self, query, trace=None):
        """Deep fused reasoning. Returns (response, used) or (None, False).

        Attempts to decompose the query into operator chains and execute
        them entirely within the embedding space.
        """
        if trace is None:
            trace = []

        # Try each reasoning pattern
        for method in [
            self._try_intersection,    # "what do X and Y have in common"
            self._try_conditional,     # "if X were Y, what would Z"
            self._try_analogy,         # "X is to Y as Z is to ?"
            self._try_chain,           # multi-step operator chains
            self._try_decompose_and_fuse,  # general: decompose → operate → compose
        ]:
            result = method(query, trace)
            if result is not None:
                return result, True

        return None, False

    def _try_intersection(self, query, trace):
        """Find what two concepts share via subspace intersection.

        "What do black holes and compression have in common?"
        → embed both → find shared high-activation dimensions → decode

        The shared dimensions ARE the common ground.
        """
        match = re.search(
            r'what\s+do\s+(.+?)\s+and\s+(.+?)\s+have\s+in\s+common',
            query.lower())
        if not match:
            match = re.search(
                r'(?:similarity|connection|relationship)\s+between\s+(.+?)\s+and\s+(.+)',
                query.lower())
        if not match:
            return None

        concept_a = match.group(1).strip().rstrip('?.')
        concept_b = match.group(2).strip().rstrip('?.')

        ea = self.embeddings.embed(concept_a, self.tokenizer)
        eb = self.embeddings.embed(concept_b, self.tokenizer)
        if ea is None or eb is None:
            return None

        # Find shared dimensions: where BOTH concepts have high activation
        # Normalize to unit vectors first
        ea_n = ea / (np.linalg.norm(ea) + 1e-10)
        eb_n = eb / (np.linalg.norm(eb) + 1e-10)

        # Element-wise minimum of absolute activations = shared signal
        shared = np.minimum(np.abs(ea_n), np.abs(eb_n))
        # Weight by sign agreement (same sign = genuinely shared)
        sign_agreement = np.sign(ea_n) * np.sign(eb_n)
        shared_signed = shared * np.maximum(sign_agreement, 0)

        # The intersection embedding: project both onto shared dimensions
        # and average
        intersection = (ea_n + eb_n) / 2.0
        # Zero out dimensions where they disagree
        intersection = intersection * (sign_agreement > 0).astype(np.float32)
        norm = np.linalg.norm(intersection)
        if norm < 0.01:
            trace.append(f"[DeepFusion] No significant overlap between "
                         f"'{concept_a}' and '{concept_b}'")
            return None

        intersection = intersection / norm

        # Decode: find words closest to the intersection embedding
        shared_words = self._decode_concept(intersection, exclude={
            concept_a, concept_b} | set(concept_a.split()) | set(concept_b.split()))

        if shared_words:
            # Also find the shared PROPERTY by looking at which corpus entries
            # are close to the intersection
            q_t = torch.tensor(intersection, dtype=torch.float32, device=DEVICE)
            scores = self.corpus.q_embeds @ q_t
            top_k = min(5, len(scores))
            top_scores, top_idx = scores.topk(top_k)

            # Extract key phrases from top matches
            shared_concepts = []
            for idx in top_idx:
                idx_val = idx.item()
                if idx_val < len(self.corpus.entries):
                    resp = self.corpus.extract_response(self.corpus.entries[idx_val])
                    if resp and len(resp) > 10:
                        # Take first sentence
                        first = resp.split('.')[0].strip()
                        if len(first) > 10 and len(first) < 200:
                            shared_concepts.append(first)

            # Build response
            overlap = float(shared_signed.sum()) / self.dim
            trace.append(f"[DeepFusion] Intersection of '{concept_a}' and "
                         f"'{concept_b}': overlap={overlap:.3f}, "
                         f"shared={shared_words[:5]}")

            if shared_concepts:
                # Use the closest corpus entry to the intersection as base
                response = (f"Both {concept_a} and {concept_b} relate to "
                            f"{', '.join(shared_words[:3])}. {shared_concepts[0]}.")
                return response
            else:
                return (f"Both {concept_a} and {concept_b} share concepts of "
                        f"{', '.join(shared_words[:4])}.")

        return None

    def _try_conditional(self, query, trace):
        """Conditional reasoning via operator chaining.

        "If photosynthesis worked in reverse, what would it produce?"
        → embed(photosynthesis) → apply(reverse/negate operator) → decode outputs

        "If gravity were stronger, what would happen?"
        → embed(gravity) → apply(intensify operator) → decode effects
        """
        match = re.search(
            r'if\s+(.+?)\s+(?:worked?|were|was|is)\s+(?:in\s+)?'
            r'(reverse|stronger|weaker|opposite|different|faster|slower)'
            r'.*?(?:what\s+would|what\s+happens|what\s+could)',
            query.lower())
        if not match:
            return None

        subject = match.group(1).strip()
        modifier = match.group(2).strip()

        emb = self.embeddings.embed(subject, self.tokenizer)
        if emb is None:
            return None

        # Map modifier to operator
        op_map = {
            'reverse': 'negate',
            'opposite': 'negate',
            'stronger': 'intensify',
            'weaker': 'diminish',
            'faster': 'intensify',
            'slower': 'diminish',
            'different': 'negate',
        }

        op_name = op_map.get(modifier)
        if not op_name:
            return None

        # Apply operator (use negate as base for most transformations)
        if op_name in ('intensify', 'diminish'):
            # Intensify: move further from origin in same direction
            # Diminish: move toward origin
            scale = 1.5 if op_name == 'intensify' else 0.5
            result_emb = emb * scale
        elif op_name == 'negate' and 'negate' in self.operators.operators:
            neg_op = self.operators.operators['negate']
            projection = np.dot(emb, neg_op)
            result_emb = emb + neg_op * abs(projection) * 2.0
        else:
            return None

        result_emb = result_emb / (np.linalg.norm(result_emb) + 1e-10)

        # Decode: what does the modified concept look like?
        # Find closest corpus entries to the transformed embedding
        q_t = torch.tensor(result_emb.astype(np.float32), device=DEVICE)
        scores = self.corpus.q_embeds @ q_t
        top_scores, top_idx = scores.topk(min(3, len(scores)))

        responses = []
        for idx in top_idx:
            idx_val = idx.item()
            if idx_val < len(self.corpus.entries):
                resp = self.corpus.extract_response(self.corpus.entries[idx_val])
                if resp and len(resp) > 20:
                    responses.append(resp)

        if responses:
            # The transformed embedding landed near these concepts
            modified_words = self._decode_concept(result_emb, exclude=set(subject.split()))
            trace.append(f"[DeepFusion] Conditional: {subject} → {modifier} → "
                         f"lands near: {modified_words[:5]}")

            # Build response from the concept the transformation landed on
            base = responses[0]
            if len(base) > 300:
                base = base[:300].rsplit('.', 1)[0] + '.'

            return (f"If {subject} worked in {modifier}, the result would relate to "
                    f"{', '.join(modified_words[:3])}. {base}")

        return None

    def _try_analogy(self, query, trace):
        """Analogy completion via parallelogram rule.

        "X is to Y as Z is to ?"
        → embed(Y) - embed(X) + embed(Z) = embed(?)

        The classic Word2Vec operation, but used for reasoning.
        """
        match = re.search(
            r'(\w+)\s+is\s+to\s+(\w+)\s+as\s+(\w+)\s+is\s+to\s+(?:what|\?)',
            query.lower())
        if not match:
            # Also try: "if X is Y, then Z is ?"
            match = re.search(
                r'if\s+(\w+)\s+is\s+(\w+)\s*,?\s+(?:then\s+)?(\w+)\s+is\s+(?:what|\?)',
                query.lower())
        if not match:
            return None

        a, b, c = match.group(1), match.group(2), match.group(3)

        ea = self.embeddings.embed(a, self.tokenizer)
        eb = self.embeddings.embed(b, self.tokenizer)
        ec = self.embeddings.embed(c, self.tokenizer)

        if ea is None or eb is None or ec is None:
            return None

        # Parallelogram: ? = B - A + C
        result_emb = eb - ea + ec
        result_emb = result_emb / (np.linalg.norm(result_emb) + 1e-10)

        # Decode
        decoded = self._decode_concept(result_emb, exclude={a, b, c})
        if decoded:
            answer = decoded[0]
            trace.append(f"[DeepFusion] Analogy: {a}:{b} :: {c}:{answer} "
                         f"(parallelogram)")
            return f"{a.capitalize()} is to {b} as {c} is to {answer}."

        return None

    def _try_chain(self, query, trace):
        """Multi-step operator chaining.

        Detect sequences of operations and apply them in order.
        "Take X, reverse it, then add Y" → chain(reverse, add)
        """
        lower = query.lower()

        # Look for explicit chaining words
        steps = re.split(r'\s*(?:then|and then|next|after that|finally)\s*', lower)
        if len(steps) < 2:
            return None

        # Extract the subject from the first step
        subject_match = re.search(r'(?:take|start with|begin with|given)\s+(.+)', steps[0])
        if not subject_match:
            return None

        current_text = subject_match.group(1).strip().rstrip('.,')
        current_emb = self.embeddings.embed(current_text, self.tokenizer)
        if current_emb is None:
            return None

        trace.append(f"[DeepFusion] Chain: starting with '{current_text}'")
        chain_results = [f"Start: {current_text}"]

        for step in steps[1:]:
            step = step.strip().rstrip('?.!')
            if not step:
                continue

            # Detect which operator this step wants
            applied = False

            # Negate/reverse
            if any(w in step for w in ['reverse', 'opposite', 'negate', 'flip']):
                if 'negate' in self.operators.operators:
                    neg_op = self.operators.operators['negate']
                    proj = np.dot(current_emb, neg_op)
                    current_emb = current_emb + neg_op * abs(proj) * 2.0
                    current_emb = current_emb / (np.linalg.norm(current_emb) + 1e-10)
                    decoded = self._decode_concept(current_emb, exclude=set())
                    current_text = decoded[0] if decoded else current_text
                    chain_results.append(f"Negate → {current_text}")
                    applied = True

            # Categorize
            elif any(w in step for w in ['categorize', 'classify', 'what type', 'category']):
                if 'categorize' in self.operators.operators:
                    cat_op = self.operators.operators['categorize']
                    current_emb = current_emb + cat_op
                    current_emb = current_emb / (np.linalg.norm(current_emb) + 1e-10)
                    decoded = self._decode_concept(current_emb, exclude=set())
                    current_text = decoded[0] if decoded else current_text
                    chain_results.append(f"Categorize → {current_text}")
                    applied = True

            # Add/combine with something
            elif 'add' in step or 'combine' in step or 'mix' in step:
                # Extract what to add
                add_match = re.search(r'(?:add|combine|mix)\s+(?:with\s+)?(.+)', step)
                if add_match:
                    other = add_match.group(1).strip()
                    other_emb = self.embeddings.embed(other, self.tokenizer)
                    if other_emb is not None:
                        current_emb = (current_emb + other_emb) / 2.0
                        current_emb = current_emb / (np.linalg.norm(current_emb) + 1e-10)
                        decoded = self._decode_concept(current_emb, exclude=set())
                        current_text = decoded[0] if decoded else f"{current_text}+{other}"
                        chain_results.append(f"Combine with {other} → {current_text}")
                        applied = True

            if not applied:
                trace.append(f"[DeepFusion] Chain: couldn't parse step '{step}'")

        if len(chain_results) > 1:
            # Decode final state
            final_words = self._decode_concept(current_emb, exclude=set())

            # Get corpus context for final embedding
            q_t = torch.tensor(current_emb.astype(np.float32), device=DEVICE)
            scores = self.corpus.q_embeds @ q_t
            _, top_idx = scores.topk(1)
            idx = top_idx[0].item()
            context = ""
            if idx < len(self.corpus.entries):
                resp = self.corpus.extract_response(self.corpus.entries[idx])
                if resp:
                    context = resp.split('.')[0] + '.'

            trace.append(f"[DeepFusion] Chain complete: {' → '.join(chain_results)}")
            result = f"{' → '.join(chain_results)}. "
            if final_words:
                result += f"Final concept: {', '.join(final_words[:3])}. "
            if context:
                result += context
            return result

        return None

    def _try_decompose_and_fuse(self, query, trace):
        """General understanding: decompose query into concepts,
        find the OPERATION implied by the query structure,
        apply it in embedding space, decode.

        This is the most general form — it handles queries that
        don't match any specific pattern by analyzing the query's
        own embedding geometry.
        """
        lower = query.lower()

        # Extract content words
        words = [w for w in re.findall(r'[a-z]{3,}', lower)
                 if w not in _QUERY_WORDS]
        if len(words) < 2:
            return None

        # Embed each concept
        word_embs = {}
        for w in words:
            emb = self.embeddings.embed(w, self.tokenizer)
            if emb is not None:
                word_embs[w] = emb

        if len(word_embs) < 2:
            return None

        # Detect the query's INTENT from its structure
        # The intent is the relationship between the concepts
        query_emb = self.embeddings.embed(query, self.tokenizer)
        if query_emb is None:
            return None

        # Compute the "intent vector": what the query adds beyond its concepts
        concept_avg = np.mean(list(word_embs.values()), axis=0)
        concept_avg = concept_avg / (np.linalg.norm(concept_avg) + 1e-10)
        intent = query_emb - concept_avg
        intent_norm = np.linalg.norm(intent)

        if intent_norm < 0.05:
            return None  # Query is just its concepts, no operation implied

        intent = intent / intent_norm

        # The intent vector IS the operation. Apply it to the concept combination.
        # Result = concepts transformed by intent
        result_emb = concept_avg + intent * 0.5
        result_emb = result_emb / (np.linalg.norm(result_emb) + 1e-10)

        # Decode via corpus proximity
        q_t = torch.tensor(result_emb.astype(np.float32), device=DEVICE)
        scores = self.corpus.q_embeds @ q_t
        top_scores, top_idx = scores.topk(min(3, len(scores)))

        best_score = float(top_scores[0])
        if best_score < 0.5:
            return None  # Too far from anything known

        idx = top_idx[0].item()
        if idx >= len(self.corpus.entries):
            return None

        resp = self.corpus.extract_response(self.corpus.entries[idx])
        if not resp or len(resp) < 20:
            return None

        # Verify the response covers the query concepts
        resp_lower = resp.lower()
        coverage = sum(1 for w in words[:4] if w in resp_lower)
        if coverage < 1:
            return None  # Response doesn't relate to query concepts

        trace.append(f"[DeepFusion] Decompose+fuse: intent_norm={intent_norm:.3f}, "
                     f"concepts={list(word_embs.keys())[:4]}, "
                     f"best_match={best_score:.3f}")

        return resp

    def _decode_concept(self, emb, exclude=None, top_k=8):
        """Decode an embedding to a list of nearby concept words.

        Uses WordDecoder (whole words) if available, falls back to BPE tokens.
        """
        if exclude is None:
            exclude = set()

        full_exclude = exclude | _QUERY_WORDS | _COMMON_STOPS

        # Use whole-word decoder with morphological exclusion
        if hasattr(self, 'decoder') and self.decoder is not None:
            return self.decoder.decode_words(emb, top_k=top_k, exclude=full_exclude,
                                              exclude_morphological=exclude)

        # Fallback: BPE token decoding
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


_QUERY_WORDS = frozenset({
    'what', 'how', 'why', 'who', 'where', 'when', 'which', 'does',
    'would', 'could', 'should', 'will', 'can', 'have', 'has', 'had',
    'tell', 'explain', 'describe', 'give', 'make', 'know', 'think',
    'like', 'about', 'between', 'common', 'happen', 'happens',
    'produce', 'create', 'compare', 'difference', 'similar',
    'worked', 'reverse', 'were', 'stronger', 'weaker',
})

_COMMON_STOPS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'and', 'or', 'but', 'not', 'no', 'to', 'of', 'in', 'for',
    'on', 'at', 'by', 'it', 'he', 'she', 'we', 'they', 'this',
    'that', 'with', 'from', 'as', 'do', 'has', 'have', 'had',
    'its', 'his', 'her', 'our', 'their', 'my', 'your',
})
