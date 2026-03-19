"""
Structural composition — form × content retrieval.

Separates FORM (joke, poem, definition) from CONTENT (cats, ocean, gravity).
Finds corpus entries at the intersection of both — the entry that best
matches both the requested form and the target content.

Currently retrieval-only: returns existing corpus entries that match
form+content. Future work: adapt form-matching entries by substituting
content words to generate truly novel text.
"""
import re
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Composition:

    FORMS = {
        'joke': {'joke', 'funny', 'humor', 'laugh'},
        'poem': {'poem', 'poetry', 'verse', 'rhyme'},
        'translation': {'translate', 'spanish', 'french', 'german'},
    }

    FORM_PATTERNS = {
        'joke': r'joke|funny|humor|laugh',
        'poem': r'poem|haiku|verse|rhyme|write.*about',
        'translation': r'translate|translation|say.*in\s+(?:spanish|french|german)',
    }

    def __init__(self, embeddings, corpus, tokenizer):
        self.embeddings = embeddings
        self.corpus = corpus
        self.tokenizer = tokenizer

    def try_compose(self, query):
        """Attempt structural composition. Returns response or None."""
        lower = query.lower()

        # Detect requested form
        form = None
        for name, pattern in self.FORM_PATTERNS.items():
            if re.search(pattern, lower):
                form = name
                break

        if form is None:
            return None

        # Extract target content
        content = re.sub(
            r'tell\s+me\s+|give\s+me\s+|write\s+|make\s+|create\s+'
            r'|a\s+joke\s+about\s+|a\s+poem\s+about\s+|a\s+haiku\s+about\s+'
            r'|translate\s+|to\s+\w+(?:\s+and\s+\w+)?'
            r'|please\s+|can\s+you\s+|about\s+',
            '', lower).strip().rstrip('?.!')

        if not content or len(content) < 2:
            return None

        content_words = set(re.findall(r'[a-z]{3,}', content))

        # Find entries matching form keywords
        form_entries = set()
        for keyword in self.FORMS.get(form, set()):
            if keyword in self.corpus.word_index:
                form_entries.update(
                    ci for ci in self.corpus.word_index[keyword]
                    if ci < self.corpus.n_entries)

        # Find entries matching content keywords
        content_entries = set()
        for word in content_words:
            if word in self.corpus.word_index:
                content_entries.update(
                    ci for ci in self.corpus.word_index[word]
                    if ci < self.corpus.n_entries)

        # Direct hit: entries matching BOTH form AND content
        intersection = form_entries & content_entries
        if intersection:
            q_emb = self.embeddings.embed(query, self.tokenizer)
            if q_emb is not None:
                q_t = torch.tensor(q_emb, dtype=torch.float32, device=DEVICE)
                cand = torch.tensor(sorted(intersection), dtype=torch.long, device=DEVICE)
                scores = self.corpus.q_embeds[cand] @ q_t
                best_idx = cand[scores.argmax()].item()
                resp = self.corpus.extract_response(self.corpus.entries[best_idx])
                if resp and len(resp) > 20:
                    return self._clean(resp[:500])

        return None

    @staticmethod
    def _clean(text):
        words = text.split()
        cleaned = []
        for i, w in enumerate(words):
            if i > 0 and w.lower() == words[i - 1].lower():
                continue
            cleaned.append(w)
        text = ' '.join(cleaned)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        last_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_end > len(text) * 0.4:
            text = text[:last_end + 1]
        return re.sub(r'\s+', ' ', text).strip()
