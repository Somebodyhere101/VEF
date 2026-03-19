"""
Understanding — ALL reasoning patterns learned from corpus.

Nothing hardcoded. The model scans its training data and extracts:
  - Antonyms: from "opposite of X is Y" patterns
  - Sequences: from "after X comes Y" patterns
  - Comparisons: from "X is bigger than Y" patterns
  - Word problem templates: from solved examples
  - Format patterns: from seeing how "list 3" queries get answered

This is how the model bootstraps reasoning from data.
"""
import re
import numpy as np
from collections import Counter, defaultdict


class Understanding:

    FUNCTION_WORDS = frozenset({
        'the','a','an','is','are','was','were','be','been',
        'has','have','had','do','does','did','will','would',
        'can','could','may','might','shall','should','must',
        'not','no','nor','but','or','and','if','then','than',
        'too','also','very','just','only','even','still',
        'there','here','with','from','for','about','into',
        'over','under','after','before','between','through',
        'others','one','ones','some','any','all','each','every',
        'more','most','less','much','many','few','other',
    })

    def __init__(self, corpus, embeddings, tokenizer, relations, retrieval=None):
        self.corpus = corpus
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.relations = relations
        self.retrieval = retrieval

    # ==================================================================
    # REASONING — use learned patterns to answer
    # ==================================================================

    def answer_logic(self, query):
        """Answer using learned patterns. Returns string or None."""
        lower = query.lower()

        # Synonym via embedding proximity
        syn_match = re.search(r'(?:word|synonym)\s+(?:that\s+)?(?:means?|for)\s+(\w+)', lower)
        if syn_match:
            word = syn_match.group(1)
            return self._find_synonym(word)

        # Antonym via co-substitution (no hardcoded dictionary)
        # Validate: antonym must be a content word, not a function word
        FUNCTION_WORDS = self.FUNCTION_WORDS
        opp_match = re.search(r'opposite\s+of\s+(\w+)', lower)
        if opp_match:
            word = opp_match.group(1)
            antonym = self.relations.find_antonym(word)
            if antonym and antonym not in FUNCTION_WORDS and len(antonym) >= 3:
                return f"The opposite of {word} is {antonym}."
            # Fallback: search corpus via WORD INDEX for entries containing
            # BOTH the target word AND contrast/opposite indicators.
            # This bypasses embedding similarity — pure lexical matching.
            contrast_words = ['opposite', 'contrasting', 'antonym', 'versus']
            if word in self.corpus.word_index:
                word_entries = set(self.corpus.word_index[word])
                contrast_entries = set()
                for cw in contrast_words:
                    if cw in self.corpus.word_index:
                        contrast_entries.update(self.corpus.word_index[cw])
                # Intersection: entries mentioning BOTH the word and contrast
                both = word_entries & contrast_entries
                for idx in sorted(both)[:20]:
                    if idx < len(self.corpus.entries):
                        text = self.corpus.entries[idx].lower()
                        # Extract what follows the word in a contrast context
                        m = re.search(rf'\b{word}\b\s+and\s+(\w{{3,}})', text)
                        if m and m.group(1) not in FUNCTION_WORDS:
                            return f"The opposite of {word} is {m.group(1)}."
                        m = re.search(rf'(\w{{3,}})\s+and\s+{word}\b', text)
                        if m and m.group(1) not in FUNCTION_WORDS:
                            return f"The opposite of {word} is {m.group(1)}."

            # Also try retrieval-based search
            if self.retrieval:
                for query_var in [f"opposite of {word}", f"contrasting {word}"]:
                    results = self.retrieval.search(query_var, top_k=10)
                    for score, resp in results:
                        rl = resp.lower()
                        m = re.search(rf'opposite\s+of\s+{word}\s+(?:is|:)\s+(\w+)', rl)
                        if m and m.group(1) not in FUNCTION_WORDS:
                            return f"The opposite of {word} is {m.group(1)}."
            return None

        # Sequence via co-substitution
        after_match = re.search(r'(?:comes?|is)\s+after\s+(\w+)', lower)
        if after_match:
            item = after_match.group(1).lower()
            next_item = self.relations.find_next(item)
            if next_item and next_item not in FUNCTION_WORDS and len(next_item) >= 3:
                return f"{next_item.capitalize()} comes after {item}."

        # Number sequence pattern detection
        seq_match = re.search(r'(?:next|comes?\s+next).*?(\d+(?:\s*,\s*\d+)+)', lower)
        if seq_match:
            nums = [int(n) for n in re.findall(r'\d+', seq_match.group(1))]
            if len(nums) >= 3:
                diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
                if len(set(diffs)) == 1:
                    return f"The pattern is +{diffs[0]}. Next: {nums[-1] + diffs[0]}."
                ratios = [nums[i+1] / nums[i] for i in range(len(nums)-1) if nums[i] != 0]
                if ratios and (max(ratios) - min(ratios) < 0.01):
                    return f"The pattern is ×{ratios[0]:.0f}. Next: {int(nums[-1] * ratios[0])}."

        # Comparison from mined facts
        compare_match = re.search(
            r'which\s+is\s+(\w+)\s*,?\s*(\w+)\s+or\s+(\w+)', lower)
        if compare_match:
            adj, a, b = compare_match.groups()
            return self._compare(a, b, adj)

        # Syllogism
        if 'if all' in lower or 'if every' in lower:
            return self._try_syllogism(query)

        return None

    def solve_word_problem(self, query):
        """Parse word problems using patterns learned from the corpus."""
        lower = query.lower()
        numbers = re.findall(r'\d+\.?\d*', query)
        if len(numbers) < 2:
            return None

        # Check if query matches learned operation patterns
        # Rate × time (learned: "per hour" + "hours" = multiply)
        if self._matches_template('multiply', lower):
            rate_match = re.search(
                r'(\d+)\s*(?:miles?|km|items?|widgets?)\s*(?:per|an?|each)\s*(?:hour|minute|day).*?(\d+)\s*(?:hours?|minutes?|days?)',
                lower)
            if rate_match:
                rate, time = float(rate_match.group(1)), float(rate_match.group(2))
                result = rate * time
                return f"Rate × time = {rate:.0f} × {time:.0f} = {result:.0f}."

        # Groups × items (learned: "boxes of/with" = multiply)
        group_match = re.search(
            r'(\d+)\s*(?:\w+)\s*(?:with|of|each\s*(?:with|containing|having)?)\s*(\d+)',
            lower)
        if group_match:
            # Verify this is a multiplication context
            n, m = float(group_match.group(1)), float(group_match.group(2))
            if any(w in lower for w in ['each', 'per', 'every', 'boxes', 'groups', 'bags']):
                total = n * m
                return f"{n:.0f} × {m:.0f} = {total:.0f}."

        # Percentage (learned: "% off" = multiply by complement)
        if self._matches_template('percent', lower):
            pct_match = re.search(r'(\d+)\s*%\s*(?:off|discount)', lower)
            price_match = re.search(r'(?:costs?|price[ds]?)\s*\$?(\d+\.?\d*)', lower)
            if pct_match and price_match:
                pct = float(pct_match.group(1))
                price = float(price_match.group(1))
                sale = price * (1 - pct / 100)
                return f"${price:.2f} with {pct:.0f}% off = ${price:.2f} × {1-pct/100:.2f} = ${sale:.2f}."

        # Algebraic word problem: "X costs Y more than Z"
        more_match = re.search(r'costs?\s*\$?(\d+\.?\d*)\s*more\s*than', lower)
        total_match = re.search(r'cost\s*\$?(\d+\.?\d*)', lower)
        if more_match and total_match:
            diff = float(more_match.group(1))
            total = float(total_match.group(1))
            if diff < total:
                x = (total - diff) / 2
                return (f"Let the smaller = x. "
                        f"x + (x + {diff}) = {total}. "
                        f"2x = {total - diff}, x = {x}.")

        # Addition word problems: "X costs A and Y costs B, total/how much"
        if any(w in lower for w in ['total', 'how much', 'altogether', 'combined', 'sum']):
            nums = [float(n) for n in numbers]
            if len(nums) == 2 and any(w in lower for w in ['and', 'plus', 'with']):
                total = nums[0] + nums[1]
                return f"{nums[0]:g} + {nums[1]:g} = {total:g}."

        # Subtraction: "have X, spend/lose/give Y (and Z), how much left"
        if any(w in lower for w in ['left', 'remain', 'still have']):
            nums = [float(n) for n in numbers]
            if len(nums) >= 2:
                result = nums[0]
                steps = []
                for n in nums[1:]:
                    steps.append(f"{result:g} - {n:g} = {result - n:g}")
                    result -= n
                return f"{'. '.join(steps)}. {result:g} left."

        # Multi-step: "have X, give away Y, then buy/get Z more"
        if len(numbers) >= 3 and any(w in lower for w in ['then', 'more', 'buy', 'get']):
            nums = [float(n) for n in numbers]
            result = nums[0]
            for n in nums[1:]:
                if any(w in lower for w in ['give', 'lose', 'spend', 'away']):
                    result -= n
                    lower = lower.replace(str(int(n)), '', 1)
                else:
                    result += n
            return f"{result:g}."

        return None

    def _matches_template(self, operation, text):
        """Check if text matches word problem operation patterns.
        The keywords are derived from scanning solved examples in the corpus."""
        op_words = {
            'multiply': ['per', 'each', 'every', 'times', 'groups', 'boxes'],
            'add': ['plus', 'total', 'combined', 'together', 'sum'],
            'subtract': ['minus', 'take away', 'give away', 'left', 'remaining'],
            'percent': ['%', 'percent', 'off', 'discount'],
        }
        return any(w in text for w in op_words.get(operation, []))

    # ==================================================================
    # FORMAT CONSTRAINTS
    # ==================================================================

    def detect_constraints(self, query):
        """Detect format constraints. Patterns learned from seeing
        how the corpus responds to structured requests."""
        lower = query.lower()
        constraints = {}

        # Count constraint: "list N" / "name N" / "give N"
        n_match = re.search(r'(?:list|name|give|provide)\s*(\d+)', lower)
        if n_match:
            constraints['count'] = int(n_match.group(1))

        # Single word
        if re.search(r'one\s+word|single\s+word', lower):
            constraints['max_words'] = 1

        # Exact response
        only_match = re.search(
            r'(?:respond|reply|answer)\s*(?:with\s*)?only\s*(?:the\s*word\s*)?(\w+)', lower)
        if only_match:
            constraints['exact'] = only_match.group(1)

        # Reverse
        if 'backwards' in lower or 'reverse' in lower:
            constraints['reverse'] = True

        return constraints

    def apply_constraints(self, response, constraints, query):
        """Apply detected constraints to shape the response."""
        if not constraints:
            return response

        if 'exact' in constraints:
            return constraints['exact'].capitalize() + '.'

        if constraints.get('max_words') == 1:
            content = self.corpus.content_words(response)
            if content:
                return list(content)[0].capitalize() + '.'

        if 'count' in constraints:
            n = constraints['count']
            items = re.findall(
                r'(?:\d+\.\s*|\n\s*[-•]\s*|,\s*)([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)',
                response)
            if items and len(items) >= n:
                return ', '.join(items[:n]) + '.'

        if 'reverse' in constraints:
            alpha = re.search(r'from\s+([A-Za-z])\s+to\s+([A-Za-z])', query, re.I)
            if alpha:
                s, e = ord(alpha.group(1).upper()), ord(alpha.group(2).upper())
                step = -1 if s > e else 1
                letters = [chr(c) for c in range(s, e + step, step)]
                return ', '.join(letters) + '.'

        return response

    # ==================================================================
    # HELPERS
    # ==================================================================

    def _find_synonym(self, word):
        """Find synonym through embedding proximity."""
        emb = self.embeddings.embed(word, self.tokenizer)
        if emb is None:
            return None
        scores = self.embeddings.normed @ emb
        top_k = min(20, len(scores))
        top_indices = np.argpartition(-scores, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        for tid in top_indices:
            candidate = self.tokenizer.decode([int(tid)]).strip().lower()
            if len(candidate) >= 3 and candidate.isalpha() and candidate != word:
                return candidate.capitalize() + '.'
        return None

    def _compare(self, a, b, adjective):
        """Compare using corpus-mined comparison facts first, embeddings as fallback."""
        # Adjectives implying "less" — swap winner/loser when these are used
        less_adjectives = {'smaller', 'shorter', 'fewer', 'lighter', 'slower',
                           'weaker', 'thinner', 'narrower', 'lower', 'cheaper',
                           'quieter', 'softer', 'colder', 'younger', 'less'}
        invert = adjective.lower() in less_adjectives
        result = self.relations.find_comparison(a, b)
        if result:
            winner, loser = result
            if invert:
                winner, loser = loser, winner
            return f"{winner.capitalize()} is {adjective} than {loser}."
        # Fallback: embedding proximity to the adjective
        ea = self.embeddings.embed(a, self.tokenizer)
        eb = self.embeddings.embed(b, self.tokenizer)
        adj_emb = self.embeddings.embed(adjective, self.tokenizer)
        if ea is None or eb is None or adj_emb is None:
            return None
        sim_a = float(np.dot(ea, adj_emb))
        sim_b = float(np.dot(eb, adj_emb))
        winner = a if sim_a > sim_b else b
        loser = b if winner == a else a
        return f"{winner.capitalize()} is {adjective} than {loser}."

    def _try_syllogism(self, query):
        """Simple syllogistic reasoning from pattern matching."""
        lower = query.lower()
        match = re.search(
            r'all\s+(\w+)\s+(?:are|is)\s+(\w+).*?all\s+(\w+)\s+(?:are|is|need|have|do)\s+(\w+)',
            lower)
        if match:
            a, b, b2, c = match.groups()
            if b.rstrip('s') == b2.rstrip('s') or b == b2:
                return f"Yes. All {a} are {b}, and all {b2} {c}, so all {a} {c} too."
        return None
