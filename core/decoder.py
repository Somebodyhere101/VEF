"""
Whole-Word Decoder — bridge from embedding space to language.

BPE tokens are subwords: "hematics", "ynthesis", "ier". 
The basis computes correctly but decodes to fragments.

Solution: build a whole-word vocabulary with embeddings,
decode to nearest whole word, not nearest subword.

Construction:
  1. Scan corpus for all unique words (3+ chars, alphabetic)
  2. Embed each word using IDF-weighted token average
  3. Store as a (N_words, dim) matrix
  4. Decode = nearest-neighbor in this matrix
"""
import re
import re
import os
import pickle
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WordDecoder:
    """Decode embeddings to whole words, not BPE fragments."""

    def __init__(self, embeddings, corpus, tokenizer, data_dir=None):
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.dim = embeddings.dim

        # Try cache
        if data_dir:
            cache_path = os.path.join(data_dir, 'word_decoder.pkl')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                self.words = cached['words']
                self.word_embeds = cached['word_embeds']
                self.word_embeds_t = torch.tensor(
                    self.word_embeds, dtype=torch.float32, device=DEVICE)
                # Build word→index map
                self.word_to_idx = {w: i for i, w in enumerate(self.words)}
                print(f"  Word decoder: {len(self.words)} words (cached)")
                return

        print("  Building whole-word decoder...")

        # Collect unique words from corpus
        word_counts = {}
        for entry in corpus.entries:
            for w in re.findall(r'[a-zA-Z]{3,}', entry):
                w_lower = w.lower()
                word_counts[w_lower] = word_counts.get(w_lower, 0) + 1

        # Filter aggressively for real English words
        # Strategy: frequency + linguistic heuristics + awareness check
        raw_candidates = [
            w for w, c in word_counts.items()
            if c >= 15 and w.isalpha() and 3 <= len(w) <= 15
            and w not in _STOP_WORDS
            and not any(ch.isupper() for ch in w)
            and not _looks_like_code(w)
            and _has_english_structure(w)
        ]

        # Second pass: use awareness to filter — real English words have
        # high basis energy and low concentration (distributed meaning)
        from core.awareness import Awareness
        _aw = Awareness(embeddings, tokenizer)
        self.words = []
        for w in sorted(raw_candidates):
            cat, energy, conc, _ = _aw.measure(w)
            if cat in ('meaning', 'partial') and energy > 0.7:
                self.words.append(w)
        del _aw

        print(f"    Vocabulary: {len(self.words)} whole words")

        # Embed each word
        embeds = []
        valid_words = []
        batch_size = 500
        for i in range(0, len(self.words), batch_size):
            batch = self.words[i:i+batch_size]
            for w in batch:
                emb = embeddings.embed(w, tokenizer)
                if emb is not None:
                    embeds.append(emb)
                    valid_words.append(w)

        self.words = valid_words
        self.word_embeds = np.array(embeds, dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.word_embeds, axis=1, keepdims=True)
        self.word_embeds = self.word_embeds / np.maximum(norms, 1e-10)

        self.word_embeds_t = torch.tensor(
            self.word_embeds, dtype=torch.float32, device=DEVICE)
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}

        print(f"    Embedded: {len(self.words)} words")

        # Cache
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            with open(os.path.join(data_dir, 'word_decoder.pkl'), 'wb') as f:
                pickle.dump({
                    'words': self.words,
                    'word_embeds': self.word_embeds,
                }, f)

    def decode(self, emb, top_k=5, exclude=None, exclude_morphological=None):
        """Decode an embedding to the nearest whole words.

        Args:
            exclude: set of exact words to skip
            exclude_morphological: set of words whose morphological
                relatives should also be excluded (shared stems, substrings)

        Returns list of (word, similarity) tuples.
        """
        if exclude is None:
            exclude = set()
        if exclude_morphological is None:
            exclude_morphological = set()

        emb_n = emb / (np.linalg.norm(emb) + 1e-10)
        emb_t = torch.tensor(emb_n, dtype=torch.float32, device=DEVICE)

        scores = self.word_embeds_t @ emb_t
        k = min(top_k + len(exclude) + 50, len(self.words))
        top_scores, top_idx = scores.topk(k)

        results = []
        for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
            word = self.words[idx]
            if word in exclude:
                continue
            # Skip morphological relatives
            if exclude_morphological and _is_morphological_relative(word, exclude_morphological):
                continue
            results.append((word, score))
            if len(results) >= top_k:
                break

        return results

    def decode_clean(self, emb, source_words, top_k=5, exclude=None):
        """Decode with automatic morphological exclusion of source words.

        This is the preferred method for operator decoding — it ensures
        the result is semantically different from the input, not just
        a morphological variant.
        """
        if exclude is None:
            exclude = set()
        return self.decode(emb, top_k=top_k,
                           exclude=exclude | set(source_words),
                           exclude_morphological=set(source_words))

    def decode_words(self, emb, top_k=5, exclude=None, exclude_morphological=None):
        """Convenience: return just the words."""
        return [w for w, s in self.decode(emb, top_k, exclude, exclude_morphological)]

    def decode_best(self, emb, exclude=None, exclude_morphological=None):
        """Return the single best word."""
        results = self.decode(emb, top_k=1, exclude=exclude,
                              exclude_morphological=exclude_morphological)
        return results[0][0] if results else None

    def decode_clean_words(self, emb, source_words, top_k=5, exclude=None):
        """Convenience: decode_clean returning just words."""
        return [w for w, s in self.decode_clean(emb, source_words, top_k, exclude)]

    def decode_clean_best(self, emb, source_words, exclude=None):
        """Convenience: decode_clean returning single best word."""
        results = self.decode_clean(emb, source_words, top_k=1, exclude=exclude)
        return results[0][0] if results else None

    def decode_number(self, emb):
        """Decode to the nearest number word or digit string."""
        # Check against number words and digit strings
        best_num = None
        best_sim = -1

        emb_n = emb / (np.linalg.norm(emb) + 1e-10)

        for n in range(-50, 201):
            n_str = str(n)
            if n_str in self.word_to_idx:
                idx = self.word_to_idx[n_str]
                sim = float(np.dot(emb_n, self.word_embeds[idx]))
                if sim > best_sim:
                    best_sim = sim
                    best_num = n
            else:
                # Fall back to token embedding
                n_emb = self.embeddings.embed(n_str, self.tokenizer)
                if n_emb is not None:
                    n_emb_n = n_emb / (np.linalg.norm(n_emb) + 1e-10)
                    sim = float(np.dot(emb_n, n_emb_n))
                    if sim > best_sim:
                        best_sim = sim
                        best_num = n

        # Also check number words in the vocabulary
        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'twenty': 20,
            'thirty': 30, 'forty': 40, 'fifty': 50, 'hundred': 100,
        }
        for word, val in number_words.items():
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                sim = float(np.dot(emb_n, self.word_embeds[idx]))
                if sim > best_sim:
                    best_sim = sim
                    best_num = val

        if best_num is not None and best_sim > 0.3:
            return best_num
        return None


_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'and', 'or', 'but', 'not', 'no', 'to', 'of', 'in', 'for',
    'on', 'at', 'by', 'it', 'he', 'she', 'we', 'they', 'this',
    'that', 'with', 'from', 'as', 'do', 'has', 'have', 'had',
    'its', 'his', 'her', 'our', 'their', 'my', 'your', 'you',
    'will', 'would', 'can', 'could', 'may', 'might', 'shall',
    'should', 'must', 'been', 'being', 'did', 'does',
})

# Code-like patterns that leak through from programming corpora
_CODE_PATTERNS = re.compile(
    r'^(?:str|int|bool|float|var|def|cls|obj|ptr|buf|src|dst|tmp|'
    r'arg|ret|idx|len|std|ctx|cfg|msg|err|req|res|ref|val|'
    r'init|impl|func|enum|void|null|true|false|char|byte|'
    r'http|html|json|xml|css|php|sql|api|url|img|div|'
    r'nalas|dwm|dwg|dwc|pardo|swiftmodule|cppclass|'
    r'uint|sint|ulong|ifdef|endif|ifndef|pragma|'
    r'concat|substr|strlen|malloc|calloc|realloc|printf|scanf|'
    r'argv|argc|stdin|stdout|stderr)$'
)

def _is_morphological_relative(word, source_words):
    """Check if word is a morphological relative of any source word.

    Catches: fast→fasten, dog→mydog, piano→pino, king→kingdom
    """
    for src in source_words:
        src = src.lower()
        # Substring/superstring
        if src in word or word in src:
            return True
        # Shared prefix (>60% of shorter word)
        min_len = min(len(word), len(src))
        shared = 0
        for a, b in zip(word, src):
            if a == b:
                shared += 1
            else:
                break
        if shared > min_len * 0.6 and shared >= 3:
            return True
        # Shared suffix (>50% of shorter word)
        shared_suffix = 0
        for a, b in zip(reversed(word), reversed(src)):
            if a == b:
                shared_suffix += 1
            else:
                break
        if shared_suffix > min_len * 0.5 and shared_suffix >= 3:
            return True
        # Edit distance <= 2 for short words
        if min_len <= 6 and _edit_distance(word, src) <= 2:
            return True
    return False

def _edit_distance(a, b):
    """Simple edit distance."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
        prev = curr
    return prev[len(b)]

def _looks_like_code(word):
    """Detect words that are likely code tokens, not English."""
    if _CODE_PATTERNS.match(word):
        return True
    # Too many consonants in a row = likely abbreviation/code
    if re.search(r'[bcdfghjklmnpqrstvwxz]{5,}', word):
        return True
    # Ends with common file extensions
    if re.search(r'(?:cpp|hpp|java|pyc|dll|exe|lib|obj)$', word):
        return True
    return False

def _has_english_structure(word):
    """Check if a word has plausible English phonotactics."""
    # Must contain at least one vowel
    if not re.search(r'[aeiouy]', word):
        return False
    # Vowel ratio: English words are typically 30-60% vowels
    vowels = sum(1 for c in word if c in 'aeiouy')
    ratio = vowels / len(word)
    if ratio < 0.15 or ratio > 0.75:
        return False
    # No triple consonants that aren't common English clusters
    if re.search(r'[bcdfghjklmnpqrstvwxz]{4,}', word):
        # Allow common clusters: str, thr, scr, spl, etc.
        cleaned = re.sub(r'(?:str|thr|scr|spl|spr|ght|ngs|nks|rch|tch)', '', word)
        if re.search(r'[bcdfghjklmnpqrstvwxz]{4,}', cleaned):
            return False
    # No common web/code compound patterns
    if re.search(r'(?:http|www|com|org|net|data|info|wiki|blog|tech|'
                 r'module|class|func|type|node|server|client|config)', word):
        return False
    return True
