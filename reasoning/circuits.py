"""
Circuits — specialized reasoning pathways.

Specialized reasoning pathways:
  - Multi-hop: separates intent from concept, retrieves both, composes
  - Self-edit: reads own output, scores coherence, replaces weak chunks
  - Concept properties: mines associated words from corpus
  - Category lookup: finds category members via embedding similarity
"""
import re
import os
import pickle
import numpy as np


class Circuits:
    """Reasoning circuits from the standalone VEF model."""

    def __init__(self, embeddings, corpus, tokenizer, retrieval):
        self.embeddings = embeddings
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.retrieval = retrieval

        # Load mined categories
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        self.categories = {}
        cat_path = os.path.join(data_dir, 'categories_clean.pkl')
        if os.path.exists(cat_path):
            with open(cat_path, 'rb') as f:
                self.categories = pickle.load(f)

    # ══════════════════════════════════════════════════════════
    # MULTI-HOP RETRIEVAL — the core circuit from standalone VEF
    # Separates intent from concept, retrieves both, composes
    # ══════════════════════════════════════════════════════════

    def multi_hop(self, query, trace=None):
        """Multi-hop compositional retrieval.

        Hop 1: Direct retrieval (full query)
        Hop 2: Concept-focused retrieval (content words only)
        Hop 3: Compose — find responses matching BOTH intent AND concept

        Returns best response or None.
        """
        if trace is None:
            trace = []

        # Separate intent words (structural, common) from concept words (content, rare)
        structural_threshold = np.log(20)
        intent_words, concept_words = [], []
        for w in re.findall(r'[a-z]+', query.lower()):
            if len(w) < 3:
                continue
            if w not in self.corpus.word_index:
                concept_words.append(w)
                continue
            n_docs = len(self.corpus.word_index[w])
            word_idf = np.log(self.corpus.n_entries / max(n_docs, 1))
            if word_idf <= structural_threshold:
                intent_words.append(w)
            else:
                concept_words.append(w)

        # Hop 1: Direct retrieval
        results = self.retrieval.search(query, top_k=20)
        if not results:
            return None

        best_resp = results[0][1]
        best_score = results[0][0]

        # Hop 2: Check concept coverage
        if concept_words:
            resp_lower = best_resp.lower()
            coverage = sum(1 for w in concept_words if w in resp_lower)
            ratio = coverage / len(concept_words)

            if ratio < 0.5:
                # Best response doesn't contain key concepts — do concept-focused hop
                concept_query = ' '.join(concept_words)
                concept_results = self.retrieval.search(concept_query, top_k=20)

                if concept_results:
                    # Hop 3: Compose — score by intent AND concept match
                    intent_emb = self.embeddings.embed(' '.join(intent_words), self.tokenizer) if intent_words else None
                    best_composed = None
                    best_composed_score = -1

                    for score, resp in concept_results:
                        rl = resp.lower()
                        concept_match = sum(1 for w in concept_words if w in rl) / len(concept_words)
                        intent_match = sum(1 for w in intent_words if w in rl) / max(len(intent_words), 1)

                        # Intent embedding alignment
                        intent_sim = 0.5
                        if intent_emb is not None:
                            r_emb = self.embeddings.embed(resp[:200], self.tokenizer)
                            if r_emb is not None:
                                intent_sim = float(np.dot(intent_emb, r_emb))

                        composed = score * (1 + concept_match) * (1 + intent_sim)
                        if composed > best_composed_score:
                            best_composed_score = composed
                            best_composed = resp

                    if best_composed:
                        composed_coverage = sum(1 for w in concept_words if w in best_composed.lower())
                        # Only use composed if it covers MORE concepts AND is substantial
                        if composed_coverage > coverage and len(best_composed) > 30:
                            best_resp = best_composed
                            trace.append(f"[Multi-hop] Concept-refined: coverage {coverage}→{composed_coverage}")

        # Return best response only if it's substantial
        if best_resp and len(best_resp) > 15:
            return best_resp
        return None

    # ══════════════════════════════════════════════════════════
    # CONCEPT PROPERTIES — mine associated words for any concept
    # ══════════════════════════════════════════════════════════

    def concept_properties(self, concept, max_entries=500):
        """Get the top associated words for a concept from the corpus."""
        from collections import Counter
        word_counts = Counter()
        concept_lower = concept.lower()
        stop = {'the','is','a','an','and','or','of','in','to','for','it','that',
                'this','are','was','with','as','be','on','at','by','not','from',
                'but','have','they','we','you','can','will','has','its','their',
                'more','also','been','which','than','other','may','had','each',
                'about','would','into','some','what','how','why','who','when',
                'where','there','all','one','like','them','these','such','do',
                'does','did','than','then','just','very','because','your','our'}

        count = 0
        for text in self.corpus.entries:
            if concept_lower in text.lower():
                resp = self.corpus.extract_response(text)
                words = set(re.findall(r'[a-z]{3,}', resp.lower())) - stop - {concept_lower}
                for w in words:
                    word_counts[w] += 1
                count += 1
                if count >= max_entries:
                    break

        return [w for w, _ in word_counts.most_common(20)]

    # ══════════════════════════════════════════════════════════
    # SELF-EDIT CIRCUIT — reads own output, fixes weak spots
    # ══════════════════════════════════════════════════════════

    def self_edit(self, draft, concept, max_passes=3):
        """Self-editing loop — the model reads its own output and fixes it.

        Each pass:
          1. Score each chunk by coherence with the whole
          2. Find the weakest chunk
          3. Replace it with a better corpus fragment
        """
        current = draft

        for _ in range(max_passes):
            full_emb = self.embeddings.embed(current[:200], self.tokenizer)
            if full_emb is None:
                break

            # Split into chunks
            phrases = re.split(r'([.!?,;])', current)
            chunks = []
            for i in range(0, len(phrases) - 1, 2):
                chunks.append(phrases[i] + (phrases[i+1] if i+1 < len(phrases) else ''))
            if len(phrases) % 2 == 1:
                chunks.append(phrases[-1])
            chunks = [c.strip() for c in chunks if len(c.strip()) > 3]

            if len(chunks) < 2:
                break

            # Score each chunk by coherence
            chunk_scores = []
            for chunk in chunks:
                c_emb = self.embeddings.embed(chunk, self.tokenizer)
                if c_emb is not None:
                    coherence = float(np.dot(full_emb, c_emb))
                    chunk_scores.append((coherence, chunk))
                else:
                    chunk_scores.append((0.0, chunk))

            if not chunk_scores:
                break

            chunk_scores.sort(key=lambda x: x[0])
            weakest_score, weakest_chunk = chunk_scores[0]
            avg_coherence = np.mean([s for s, _ in chunk_scores])

            if weakest_score > avg_coherence * 0.7:
                break  # no weak spots

            # Replace weak chunk with better corpus fragment
            replacement = self._find_replacement(weakest_chunk, concept, full_emb)
            if replacement:
                current = current.replace(weakest_chunk, replacement, 1)

        return current

    def _find_replacement(self, weak_chunk, concept, context_emb):
        """Find a better replacement for a weak chunk from the corpus."""
        results = self.retrieval.search(f"{concept} {weak_chunk}", top_k=10)
        if not results:
            return None

        best_replacement = None
        best_coherence = -1

        for score, resp in results:
            # Extract chunks from this response
            resp_chunks = [c.strip() for c in re.split(r'[.!?]', resp) if len(c.strip()) > 10]
            for chunk in resp_chunks[:3]:
                c_emb = self.embeddings.embed(chunk, self.tokenizer)
                if c_emb is not None:
                    coherence = float(np.dot(context_emb, c_emb))
                    if coherence > best_coherence and chunk.lower() != weak_chunk.lower():
                        best_coherence = coherence
                        best_replacement = chunk

        return best_replacement

    # ══════════════════════════════════════════════════════════
    # CATEGORY CIRCUIT — list members via embedding similarity
    # ══════════════════════════════════════════════════════════

    def category_lookup(self, query):
        """Find category members using mined data + embedding scoring."""
        q_words = [w.lower() for w in query.split() if len(w) >= 3]
        if not q_words:
            return None

        # Find which query word matches a known category
        best_cat, best_sim = None, -1
        for word in q_words:
            w_emb = self.embeddings.embed(word, self.tokenizer)
            if w_emb is None:
                continue
            for variant in [word, word + 's', word.rstrip('s')]:
                if variant in self.categories:
                    cat_emb = self.embeddings.embed(variant, self.tokenizer)
                    if cat_emb is not None:
                        sim = float(np.dot(w_emb, cat_emb))
                        if sim > best_sim:
                            best_sim = sim
                            best_cat = variant

        if best_cat is None or best_cat not in self.categories:
            return None

        # Score members by embedding similarity to category
        cat_emb = self.embeddings.embed(best_cat, self.tokenizer)
        if cat_emb is None:
            return None

        scored = []
        for member in self.categories[best_cat]:
            if len(member) < 3 or member == best_cat:
                continue
            m_emb = self.embeddings.embed(member, self.tokenizer)
            if m_emb is not None:
                sim = float(np.dot(cat_emb, m_emb))
                scored.append((sim, member))

        scored.sort(reverse=True)
        good = [m for s, m in scored if s > 0.2][:5]
        if good:
            return ', '.join(good).capitalize() + '.'
        return None
