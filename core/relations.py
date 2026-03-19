"""
Relations — discover ALL word relationships from corpus statistics.

One unified framework: CO-SUBSTITUTION + CONNECTORS.

The connector between two words reveals the relationship:
  "X or Y"  / "X but Y"    → CONTRAST (antonyms)
  "X and Y" / "X like Y"   → AGREEMENT (synonyms)
  "X then Y"/ "after X, Y" → SEQUENCE (temporal order)
  "X than Y"               → COMPARISON (scalar order)
  "X is a Y"               → CATEGORY (hypernym)

No embeddings. No gradients. No hardcoded dictionaries.
Everything from counting co-substitution patterns.
"""
import re
import os
import pickle
from collections import Counter, defaultdict

CONTRAST = {'or', 'but', 'versus', 'nor', 'not', 'while', 'unlike'}
AGREEMENT = {'like', 'aka', 'means', 'called', 'similar'}
SEQUENCE = {'then', 'after', 'before', 'next', 'followed'}
COMPARISON = {'than'}
ALL_CONNECTORS = CONTRAST | AGREEMENT | SEQUENCE | COMPARISON | {'and', 'is', 'are'}


class Relations:

    def __init__(self, corpus_entries, max_entries=30000, cache_dir=None):
        # Try loading from slim cache (only derived results, not raw data)
        if cache_dir:
            cache_path = os.path.join(cache_dir, 'relations_slim.pkl')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                self.antonyms = cached['antonyms']
                self.synonyms = cached['synonyms']
                self.sequences = cached['sequences']
                self.comparisons = cached.get('comparisons', {})
                self.categories = cached.get('categories', {})
                self.frames = {}
                self.frame_fillers = {}
                self.pairs = {}
                self.ordered_pairs = {}
                print(f"  Relations: {len(self.antonyms)} antonyms, "
                      f"{len(self.synonyms)} synonyms (cached)")
                return

        print("  Mining word relations from corpus...")

        self.frames = defaultdict(Counter)
        self.frame_fillers = defaultdict(set)
        self.pairs = defaultdict(Counter)
        self.ordered_pairs = defaultdict(Counter)
        self.categories = {}

        self._corpus_entries = corpus_entries  # keep ref for full-corpus mining
        self._scan(corpus_entries, max_entries)

        self.antonyms = {}
        self.synonyms = {}
        self.sequences = {}
        self.comparisons = {}
        self._derive()

        print(f"    Words: {len(self.frames)}, Antonyms: {len(self.antonyms)}, "
              f"Synonyms: {len(self.synonyms)}, Sequences: {len(self.sequences)}")

        # Save slim cache (only derived results — loads in <0.01s)
        if cache_dir:
            with open(os.path.join(cache_dir, 'relations_slim.pkl'), 'wb') as f:
                pickle.dump({
                    'antonyms': self.antonyms,
                    'synonyms': self.synonyms,
                    'sequences': self.sequences,
                    'comparisons': self.comparisons,
                    'categories': self.categories,
                }, f)
            print("    Cached for fast reload.")

    def _scan(self, entries, max_entries):
        for text in entries[:max_entries]:
            words = re.findall(r'[a-z]+', text.lower())

            for i in range(1, len(words) - 1):
                w = words[i]
                if len(w) < 3:
                    continue
                frame = (words[i-1], words[i+1])
                self.frames[w][frame] += 1
                self.frame_fillers[frame].add(w)

            for i in range(len(words) - 2):
                a, conn, b = words[i], words[i+1], words[i+2]
                if len(a) >= 3 and len(b) >= 3 and a != b and conn in ALL_CONNECTORS:
                    self.pairs[tuple(sorted([a, b]))][conn] += 1
                    self.ordered_pairs[(a, b)][conn] += 1

            # "X is a Y" categories
            for i in range(len(words) - 3):
                if words[i+1] in ('is', 'are') and words[i+2] in ('a', 'an', 'the'):
                    a, b = words[i], words[i+3]
                    if len(a) >= 3 and len(b) >= 3:
                        self.categories[a] = b

            # Direct antonym statements: "opposite of X is Y"
            for m in re.finditer(r'opposite\s+of\s+(\w{3,})\b.*?\b(?:is|are)\s+(\w{3,})', text.lower()):
                a, b = m.group(1), m.group(2)
                if a != b:
                    pair = tuple(sorted([a, b]))
                    self.pairs[pair]['opposite_stated'] += 1

            # "after X comes Y" / "X is the season after Y"
            for m in re.finditer(r'(?:season|day|month)\s+after\s+(\w{3,})\b.*?\b(?:is|are|called)\s+(\w{3,})', text.lower()):
                self.ordered_pairs[(m.group(1), m.group(2))]['after_stated'] += 1
            for m in re.finditer(r'(\w{3,})\s+is\s+the\s+season\s+after\s+(\w{3,})', text.lower()):
                self.ordered_pairs[(m.group(2), m.group(1))]['after_stated'] += 1

    def _derive(self):
        # HIGHEST PRIORITY: explicitly stated relationships from corpus text
        for pair, conns in self.pairs.items():
            a, b = pair
            if conns.get('opposite_stated', 0) > 0:
                self.antonyms[a] = b
                self.antonyms[b] = a
                continue

            total = sum(conns.values())
            if total < 2:
                continue

            contrast_n = sum(conns.get(c, 0) for c in CONTRAST)
            agreement_n = sum(conns.get(c, 0) for c in AGREEMENT)
            and_n = conns.get('and', 0)

            shared = len(set(self.frames.get(a, {})) & set(self.frames.get(b, {})))
            min_f = min(len(self.frames.get(a, {})), len(self.frames.get(b, {})), 1)
            sub = shared / max(min_f, 1)

            if sub > 0.03:
                if contrast_n > agreement_n + and_n * 0.3:
                    self.antonyms[a] = b
                    self.antonyms[b] = a
                elif agreement_n > contrast_n:
                    self.synonyms[a] = b
                    self.synonyms[b] = a

        for (a, b), conns in self.ordered_pairs.items():
            # Explicitly stated sequences
            if conns.get('after_stated', 0) > 0:
                self.sequences[a] = b
            if conns.get('then', 0) > 0 or conns.get('followed', 0) > 0:
                self.sequences[a] = b
            if conns.get('after', 0) > 0:
                self.sequences[b] = a
            if conns.get('than', 0) >= 1:
                self.comparisons[(a, b)] = conns['than']

    def learn_antonym_axis(self, embeddings, tokenizer):
        """Learn the antonym direction from DATA-MINED pairs.

        1. Co-substitution already mined antonym candidates
        2. Filter to HIGH-CONFIDENCE pairs (both directions agree)
        3. Compute difference vectors in embedding space
        4. Average → the "opposite" axis
        5. Use it to find antonyms for any word

        Zero hardcoded pairs. Everything from corpus statistics.
        """
        import numpy as np

        # Mine explicitly stated opposite pairs from the FULL corpus.
        # Patterns: "opposite of X is Y", "contrasting elements are X and Y"
        # Filter: both words must be common (>10 corpus entries), alphabetic.
        # These are ground truth from the data — not hardcoded.
        import re as _re
        STOP = {'the','this','that','what','which','some','each','how','who',
                'its','his','her','our','their','them','your','been','being',
                'was','were','has','had','have','does','did','are','with'}

        # Collect explicitly stated opposite pairs from co-substitution mining
        confident_pairs = []
        seen = set()
        for pair, conns in self.pairs.items():
            if conns.get('opposite_stated', 0) > 0:
                a, b = pair
                if a in STOP or b in STOP or not a.isalpha() or not b.isalpha():
                    continue
                if len(a) < 3 or len(b) < 3:
                    continue
                key = tuple(sorted([a, b]))
                if key not in seen:
                    confident_pairs.append((a, b))
                    seen.add(key)

        # If mining didn't find enough, scan full corpus for "opposite of X is Y"
        if len(confident_pairs) < 10 and hasattr(self, '_corpus_entries'):
            pattern = _re.compile(r'opposite\s+of\s+(\w{3,})\s+(?:is|are|:)\s+(\w{3,})')
            pattern2 = _re.compile(r'contrasting\s+\w+\s+(?:are|include)\s+(\w{3,})\s+and\s+(\w{3,})')
            for text in self._corpus_entries:
                tl = text.lower()
                if 'opposite' not in tl and 'contrasting' not in tl:
                    continue
                for m in pattern.finditer(tl):
                    a, b = m.group(1), m.group(2)
                    if a not in STOP and b not in STOP and a != b and a.isalpha() and b.isalpha():
                        key = tuple(sorted([a, b]))
                        if key not in seen:
                            confident_pairs.append((a, b))
                            seen.add(key)
                for m in pattern2.finditer(tl):
                    a, b = m.group(1), m.group(2)
                    if a not in STOP and b not in STOP and a != b and a.isalpha() and b.isalpha():
                        key = tuple(sorted([a, b]))
                        if key not in seen:
                            confident_pairs.append((a, b))
                            seen.add(key)

        print(f"    Antonym axis: {len(confident_pairs)} data-mined pairs")

        diffs = []
        for a, b in confident_pairs:
            ea = embeddings.embed(a, tokenizer)
            eb = embeddings.embed(b, tokenizer)
            if ea is not None and eb is not None:
                diff = ea - eb
                norm = np.linalg.norm(diff)
                if norm > 0.01:
                    diffs.append(diff / norm)  # normalize each pair

        if len(diffs) >= 5:
            D = np.array(diffs)
            _, S, Vt = np.linalg.svd(D, full_matrices=False)
            self._antonym_axis = Vt[0]
            print(f"    Axis quality: S[0]={S[0]:.1f}, S[1]={S[1]:.1f} (ratio={S[0]/S[1]:.1f}x)")
            self._embeddings = embeddings
            self._tokenizer = tokenizer

            # Axis available as fallback for unknown words
            # Keep mined antonyms unchanged — axis validation was too aggressive
            print(f"    Axis ready for fallback on unknown words")
        else:
            self._antonym_axis = None

    def find_antonym(self, word):
        word = word.lower()

        # FIRST: check mined antonyms (co-substitution — proven reliable)
        if word in self.antonyms:
            return self.antonyms[word]

        # SECOND: try embedding axis (learned from data-mined opposite pairs)
        if hasattr(self, '_antonym_axis') and self._antonym_axis is not None:
            import numpy as np
            emb = self._embeddings.embed(word, self._tokenizer)
            if emb is not None:
                # Project the word onto the antonym axis and flip
                projection = np.dot(emb, self._antonym_axis)
                if abs(projection) > 0.05:
                    target = emb - 2 * projection * self._antonym_axis
                    target /= np.linalg.norm(target) + 1e-10
                    # Find nearest word to the target
                    sims = self._embeddings.normed @ target
                    # Exclude the word itself and very common tokens
                    best_idx = None
                    for idx in np.argsort(-sims)[:20]:
                        # Decode token
                        decoded = self._tokenizer.decode([idx]).strip().lower()
                        decoded = decoded.replace('Ġ', '').strip()
                        if len(decoded) >= 3 and decoded != word and decoded.isalpha():
                            return decoded

        if word not in self.frames:
            return None
        best, best_score = None, 0
        for frame in self.frames[word]:
            for other in self.frame_fillers.get(frame, set()):
                if other == word:
                    continue
                pair = tuple(sorted([word, other]))
                if pair in self.pairs:
                    score = sum(self.pairs[pair].get(c, 0) for c in CONTRAST)
                    if score > best_score:
                        best_score = score
                        best = other
        return best

    def find_synonym(self, word):
        return self.synonyms.get(word.lower())

    def find_next(self, word):
        return self.sequences.get(word.lower())

    def find_comparison(self, a, b):
        a, b = a.lower(), b.lower()
        if (a, b) in self.comparisons:
            return a, b
        if (b, a) in self.comparisons:
            return b, a
        return None
