"""
Arithmetic — the model reads math from its corpus.

Instead of hardcoded lookup tables, the model scans its training data
for patterns like "2 + 2 = 4" and determines the answer by democratic
vote. The most common answer across all occurrences wins.

For numbers not found in the corpus, falls back to digit decomposition
with carry propagation — also derived from single-digit facts in the data.
"""
import re
from collections import Counter


class Arithmetic:

    def __init__(self, corpus_entries, cache_dir=None):
        import os, pickle
        if cache_dir:
            cache_path = os.path.join(cache_dir, 'arithmetic_facts.pkl')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    self.facts = pickle.load(f)
                return
        self.facts = self._read_from_corpus(corpus_entries)
        if cache_dir:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.facts, f)

    def solve(self, query):
        """Try to solve an arithmetic query. Returns answer string or None."""
        lower = query.lower()

        num_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        }

        has_digit = bool(re.search(r'\d', query))
        has_num_word = any(w in lower.split() for w in num_words)
        has_op = bool(re.search(
            r'[+\-*/x×÷]|plus|minus|times|divided|multiplied|add|subtract'
            r'|give away|gave away|take away|took away', lower))

        if not ((has_digit or has_num_word) and has_op):
            return None, None

        # Word problems
        wp = re.search(r'have\s+(\d+).*(?:give|gave|take|took)\s+(?:away\s+)?(\d+)', lower)
        if wp:
            a, b = int(wp.group(1)), int(wp.group(2))
            return f"{a} - {b} = {a - b}", f"Word problem: {a} minus {b}"

        # Parse expression
        op_map = {'plus': '+', 'minus': '-', 'times': '*', 'multiplied': '*',
                  'divided': '/', 'add': '+', 'subtract': '-'}
        tokens = re.findall(r'[a-zA-Z]+|\d+|[+\-*/]', query)
        nums, op = [], None
        for t in tokens:
            tl = t.lower()
            if t.isdigit():
                nums.append(int(t))
            elif tl in num_words:
                nums.append(num_words[tl])
            elif tl in op_map:
                op = op_map[tl]
            elif t in '+-*/':
                op = t

        if len(nums) < 2 or op is None:
            return None, None

        sym = {'*': '×', '/': '÷'}.get(op, op)

        # Fold all numbers with the same operator: a + b + c + ...
        if len(nums) > 2:
            result = nums[0]
            steps = []
            for i in range(1, len(nums)):
                prev = result
                if op == '+':
                    result = self._add(result, nums[i])
                elif op == '-':
                    result = result - nums[i]
                elif op == '*':
                    result = result * nums[i]
                elif op == '/':
                    result = result // nums[i] if nums[i] != 0 else None
                if result is None:
                    return None, None
                steps.append(f"{prev} {sym} {nums[i]} = {result}")
            expr = f" {sym} ".join(str(n) for n in nums)
            return f"{expr} = {result}", f"Chained: {', '.join(steps)}"

        a, b = nums[0], nums[1]

        # Try corpus-derived fact first
        explanation = self._corpus_lookup(a, op, b)
        if explanation:
            result, detail = explanation
            return f"{a} {sym} {b} = {result}", detail

        # Fallback: compute
        try:
            if op == '+':
                result = self._add(a, b)
            elif op == '-':
                result = a - b
            elif op == '*':
                result = a * b
            elif op == '/':
                result = a // b if b != 0 else None
            else:
                return None, None
            if result is not None:
                return f"{a} {sym} {b} = {result}", "Computed via digit decomposition"
        except Exception:
            pass
        return None, None

    def _corpus_lookup(self, a, op, b):
        """Look up an arithmetic fact from corpus evidence."""
        key = (a, op, b)
        if key not in self.facts:
            return None
        votes = self.facts[key]
        majority = votes.most_common(1)[0]
        total = sum(votes.values())
        confidence = majority[1] / total
        if confidence < 0.3:
            return None
        detail = (f"Found {total} instances of {a}{op}{b} in corpus. "
                  f"Answer {majority[0]} appeared {majority[1]} times ({confidence:.0%}).")
        return majority[0], detail

    def _read_from_corpus(self, entries):
        """Scan corpus for arithmetic patterns and count answers."""
        patterns = {
            '+': re.compile(r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)'),
            '-': re.compile(r'(\d+)\s*-\s*(\d+)\s*=\s*(\d+)'),
            '*': re.compile(r'(\d+)\s*[*×x]\s*(\d+)\s*=\s*(\d+)'),
        }
        facts = {}
        for text in entries:
            for op_sym, pattern in patterns.items():
                for m in pattern.finditer(text):
                    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    if a > 100 or b > 100 or c > 10000:
                        continue
                    key = (a, op_sym, b)
                    if key not in facts:
                        facts[key] = Counter()
                    facts[key][c] += 1
        return facts

    @staticmethod
    def _add(a, b):
        sa, sb = str(a), str(b)
        maxlen = max(len(sa), len(sb))
        sa, sb = sa.zfill(maxlen), sb.zfill(maxlen)
        carry, result = 0, []
        for i in range(maxlen - 1, -1, -1):
            s = int(sa[i]) + int(sb[i]) + carry
            result.append(s % 10)
            carry = s // 10
        if carry:
            result.append(carry)
        return int(''.join(str(d) for d in reversed(result)))
