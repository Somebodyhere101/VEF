"""
Corpus — the model's knowledge base.

Every answer the model gives traces back to a specific corpus entry.
The corpus is a list of text strings, each containing a Q-A pair.
"""
import os
import re
import pickle
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Corpus:

    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'corpus_texts.pkl'), 'rb') as f:
            self.entries = pickle.load(f)

        # Load embeddings as numpy first, transfer to GPU lazily on first use
        self._q_embs_np = np.load(os.path.join(data_dir, 'q_embeds_idf.npy'))
        self._q_embeds_gpu = None
        self.n_entries = len(self._q_embs_np)

        with open(os.path.join(data_dir, 'word_index.pkl'), 'rb') as f:
            self.word_index = pickle.load(f)

    @property
    def q_embeds(self):
        if self._q_embeds_gpu is None:
            self._q_embeds_gpu = torch.tensor(
                self._q_embs_np.astype(np.float32), device=DEVICE)
        return self._q_embeds_gpu

    def extract_response(self, text):
        """Get the response portion of a corpus entry."""
        if 'Assistant:' in text:
            return text.split('Assistant:', 1)[1].strip()
        lines = text.strip().split('\n')
        if len(lines) >= 2:
            return '\n'.join(lines[1:]).strip()
        return text.strip()

    def extract_question_and_response(self, text):
        """Get both the question and response."""
        if 'Human:' in text and 'Assistant:' in text:
            q = text.split('Human:', 1)[1].split('Assistant:', 1)[0].strip()
            r = text.split('Assistant:', 1)[1].strip()
            return q, r
        lines = text.strip().split('\n')
        if len(lines) >= 2:
            return lines[0].strip(), '\n'.join(lines[1:]).strip()
        return text[:100].strip(), text.strip()

    def content_words(self, query, structural_threshold=None):
        """Extract content words from a query using IDF separation.
        Words appearing in >5% of entries are structural, not content."""
        if structural_threshold is None:
            structural_threshold = np.log(20)

        words = set()
        for w in re.findall(r'[a-z]+', query.lower()):
            if len(w) < 2:
                continue
            if w not in self.word_index:
                continue
            n_docs = len(self.word_index[w])
            word_idf = np.log(self.n_entries / max(n_docs, 1))
            if word_idf > structural_threshold:
                words.add(w)
        return sorted(words)
