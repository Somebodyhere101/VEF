"""
Compute Basis — Python operations as embedding-space transformations.

The W* bridge between English and computation:

1. Build a corpus of (English description, Python code, execution trace, result)
2. Embed each component in the same 128d space
3. Compute W* = (H'H + λI)⁻¹H'Y mapping English → computation
4. At inference: English query → W* → computation embedding → decode to operation → execute

The model doesn't "call" Python. The model's basis includes Python.
Computation and language share the same geometric space.
"""
import re
import os
import pickle
import ast
import numpy as np
import torch
from collections import defaultdict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ComputeBasis:
    """Python computation fused into the embedding space."""

    def __init__(self, embeddings, tokenizer, data_dir=None):
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.dim = embeddings.dim

        # The operation vocabulary: each operation has an embedding
        self.op_embeddings = {}      # op_name → (dim,) vector
        self.op_templates = {}       # op_name → code template
        self.op_extractors = {}      # op_name → regex for extracting args

        # W* matrices: map from query embedding to operation+args embeddings
        self.W_op = None             # (dim, n_ops) — which operation
        self.W_args = None           # (2*dim, dim) — argument extraction
        self.op_names = []           # ordered op names matching W_op columns

        # Try cache
        if data_dir:
            cache_path = os.path.join(data_dir, 'compute_basis.pkl')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                self.op_embeddings = cached['op_embeddings']
                self.op_templates = cached['op_templates']
                self.W_op = cached.get('W_op')
                self.W_args = cached.get('W_args')
                self.op_names = cached.get('op_names', [])
                print(f"  Compute basis: {len(self.op_embeddings)} operations (cached)")
                self._build_extractors()
                return

        print("  Building compute basis...")
        self._build_operation_vocabulary()
        self._build_W_star()
        self._build_extractors()

        # Cache
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            with open(os.path.join(data_dir, 'compute_basis.pkl'), 'wb') as f:
                pickle.dump({
                    'op_embeddings': self.op_embeddings,
                    'op_templates': self.op_templates,
                    'W_op': self.W_op,
                    'W_args': self.W_args,
                    'op_names': self.op_names,
                }, f)

        print(f"  Compute basis: {len(self.op_embeddings)} operations, "
              f"W* shape={self.W_op.shape if self.W_op is not None else None}")

    def _build_operation_vocabulary(self):
        """Define the operation space — each operation gets an embedding.

        The embedding is built from the natural language descriptions
        of what each operation does, so English and computation share
        the same space.
        """
        # (operation_name, natural_language_descriptions, code_template, arg_count)
        operations = [
            ('add', [
                'add two numbers', 'sum of', 'plus', 'total',
                'what is X plus Y', 'X added to Y', 'combine numbers',
            ], '{a} + {b}', 2),
            ('subtract', [
                'subtract', 'minus', 'difference between', 'take away',
                'what is X minus Y', 'X subtracted from Y', 'remove from',
            ], '{a} - {b}', 2),
            ('multiply', [
                'multiply', 'times', 'product of', 'multiplied by',
                'what is X times Y', 'X multiplied by Y',
            ], '{a} * {b}', 2),
            ('divide', [
                'divide', 'divided by', 'ratio of', 'split into',
                'what is X divided by Y', 'X over Y',
            ], '{a} / {b}', 2),
            ('power', [
                'power', 'exponent', 'raised to', 'squared', 'cubed',
                'X to the power of Y', 'X raised to Y',
            ], '{a} ** {b}', 2),
            ('modulo', [
                'modulo', 'remainder', 'mod', 'remainder when divided',
                'X mod Y', 'remainder of X divided by Y',
            ], '{a} % {b}', 2),
            ('sqrt', [
                'square root', 'sqrt', 'root of',
                'square root of X', 'what is the root of X',
            ], '{a} ** 0.5', 1),
            ('factorial', [
                'factorial', 'factorial of', 'X factorial',
                'how many permutations', 'arrangements of',
            ], '__import__("math").factorial(int({a}))', 1),
            ('prime_check', [
                'is prime', 'prime number', 'primality',
                'is X a prime number', 'check if prime',
            ], 'all(int({a}) % i != 0 for i in range(2, int(int({a})**0.5)+1)) and int({a}) > 1', 1),
            ('fibonacci', [
                'fibonacci', 'fibonacci number', 'fibonacci sequence',
                'Xth fibonacci', 'fibonacci of X',
            ], '(lambda f,n:f(f,n))(lambda f,n:n if n<=1 else f(f,n-1)+f(f,n-2),int({a}))', 1),
            ('sort_asc', [
                'sort ascending', 'sort smallest to largest', 'arrange in order',
                'sort these numbers', 'order from low to high',
            ], 'sorted({a})', 1),
            ('sort_desc', [
                'sort descending', 'sort largest to smallest', 'reverse sort',
                'sort from high to low', 'order from largest',
            ], 'sorted({a}, reverse=True)', 1),
            ('reverse', [
                'reverse', 'reverse string', 'backwards', 'flip text',
                'reverse the word', 'spell backwards',
            ], 'str({a})[::-1]', 1),
            ('length', [
                'length', 'how long', 'how many characters', 'count characters',
                'length of the string', 'how many letters',
            ], 'len(str({a}))', 1),
            ('count_char', [
                'count occurrences', 'how many times', 'count the letter',
                'how many X in Y', 'occurrences of',
            ], 'str({b}).count(str({a}))', 2),
            ('abs_val', [
                'absolute value', 'abs', 'magnitude',
                'absolute value of X', 'distance from zero',
            ], 'abs({a})', 1),
            ('round_num', [
                'round', 'round to', 'nearest integer',
                'round X', 'round to nearest',
            ], 'round({a})', 1),
            ('max_val', [
                'maximum', 'largest', 'biggest', 'max of',
                'which is larger', 'the bigger number',
            ], 'max({a})', 1),
            ('min_val', [
                'minimum', 'smallest', 'min of',
                'which is smaller', 'the smaller number',
            ], 'min({a})', 1),
            ('convert_bin', [
                'convert to binary', 'binary of', 'in binary',
                'what is X in binary', 'binary representation',
            ], 'bin(int({a}))', 1),
            ('convert_hex', [
                'convert to hex', 'hexadecimal', 'in hex',
                'what is X in hex', 'hex representation',
            ], 'hex(int({a}))', 1),
        ]

        for op_name, descriptions, template, n_args in operations:
            # Embed the operation as the average of its descriptions
            embs = []
            for desc in descriptions:
                emb = self.embeddings.embed(desc, self.tokenizer)
                if emb is not None:
                    embs.append(emb)

            if embs:
                op_emb = np.mean(embs, axis=0)
                op_emb = op_emb / (np.linalg.norm(op_emb) + 1e-10)
                self.op_embeddings[op_name] = op_emb.astype(np.float32)
                self.op_templates[op_name] = (template, n_args)

    def _build_W_star(self):
        """Build W* — the closed-form mapping from English to operations.

        Training data: (English query embedding → operation embedding)
        W* = (H'H + λI)⁻¹H'Y

        This is THE equation. Actually used. Not a metaphor.
        """
        if not self.op_embeddings:
            return

        # Build training data from the operation descriptions
        H_rows = []  # English query embeddings
        Y_rows = []  # Operation one-hot labels
        self.op_names = sorted(self.op_embeddings.keys())
        op_to_idx = {name: i for i, name in enumerate(self.op_names)}

        # For each operation, its descriptions are training examples
        operations_data = self._get_training_data()
        for op_name, queries in operations_data.items():
            if op_name not in op_to_idx:
                continue
            idx = op_to_idx[op_name]
            for query_text in queries:
                emb = self.embeddings.embed(query_text, self.tokenizer)
                if emb is not None:
                    H_rows.append(emb)
                    label = np.zeros(len(self.op_names), dtype=np.float64)
                    label[idx] = 1.0
                    Y_rows.append(label)

        if len(H_rows) < 10:
            return

        H = np.array(H_rows, dtype=np.float64)  # (n, dim)
        Y = np.array(Y_rows, dtype=np.float64)  # (n, n_ops)

        # W* = (H'H + λI)⁻¹H'Y
        lam = 1e-2
        HtH = H.T @ H + lam * np.eye(H.shape[1])
        HtY = H.T @ Y
        self.W_op = np.linalg.solve(HtH, HtY).astype(np.float32)  # (dim, n_ops)

        print(f"    W* computed: {H.shape[0]} training examples, "
              f"{len(self.op_names)} operations")

    def _get_training_data(self):
        """Generate diverse training queries for each operation."""
        data = defaultdict(list)

        # Arithmetic with varied phrasing
        for a in [2, 3, 5, 7, 10, 15, 20]:
            for b in [2, 3, 4, 5, 8, 12]:
                data['add'].extend([
                    f'what is {a} plus {b}', f'{a} + {b}', f'add {a} and {b}',
                    f'sum of {a} and {b}', f'calculate {a} plus {b}',
                ])
                data['subtract'].extend([
                    f'what is {a} minus {b}', f'{a} - {b}', f'subtract {b} from {a}',
                    f'difference between {a} and {b}',
                ])
                data['multiply'].extend([
                    f'what is {a} times {b}', f'{a} * {b}', f'{a} multiplied by {b}',
                    f'product of {a} and {b}',
                ])
                data['divide'].extend([
                    f'what is {a} divided by {b}', f'{a} / {b}', f'divide {a} by {b}',
                ])

        for n in [2, 3, 4, 5, 8, 10, 16, 25, 64, 100, 144]:
            data['sqrt'].extend([
                f'square root of {n}', f'sqrt of {n}', f'what is the square root of {n}',
            ])

        for n in [3, 5, 7, 10, 12, 15, 20]:
            data['factorial'].extend([
                f'factorial of {n}', f'{n} factorial', f'what is {n} factorial',
            ])
            data['fibonacci'].extend([
                f'{n}th fibonacci number', f'fibonacci of {n}',
                f'what is the {n}th fibonacci',
            ])

        for n in [7, 11, 13, 17, 23, 29, 31, 37, 41, 97, 100, 15, 21]:
            data['prime_check'].extend([
                f'is {n} prime', f'is {n} a prime number', f'check if {n} is prime',
            ])

        data['sort_asc'].extend([
            'sort 5 2 8 1 9', 'sort these numbers 3 7 1 4',
            'arrange in order 9 3 5 1', 'sort ascending 8 2 6 4',
        ])
        data['sort_desc'].extend([
            'sort descending 5 2 8 1 9', 'sort from largest to smallest 3 7 1',
            'reverse sort 9 3 5 1', 'sort from high to low 8 2 6 4',
        ])

        for w in ['hello', 'world', 'python', 'test', 'algorithm', 'science']:
            data['reverse'].extend([
                f'reverse {w}', f'reverse the string {w}', f'{w} backwards',
                f'spell {w} backwards',
            ])
            data['length'].extend([
                f'length of {w}', f'how long is {w}', f'how many characters in {w}',
            ])

        for n in [-5, -3, 0, 3, 7, -12, 42]:
            data['abs_val'].extend([
                f'absolute value of {n}', f'abs of {n}',
            ])

        for n in [10, 255, 42, 1024]:
            data['convert_bin'].extend([
                f'convert {n} to binary', f'{n} in binary', f'binary of {n}',
            ])
            data['convert_hex'].extend([
                f'convert {n} to hex', f'{n} in hexadecimal', f'hex of {n}',
            ])

        return data

    def _build_extractors(self):
        """Build argument extraction patterns for each operation."""
        self.op_extractors = {
            'add': (re.compile(r'(-?\d+\.?\d*)\s*(?:\+|plus|added?\s+to)\s*(-?\d+\.?\d*)'), 2),
            'subtract': (re.compile(r'(-?\d+\.?\d*)\s*(?:-|minus)\s*(-?\d+\.?\d*)'), 2),
            'multiply': (re.compile(r'(-?\d+\.?\d*)\s*(?:\*|×|times|multiplied\s+by)\s*(-?\d+\.?\d*)'), 2),
            'divide': (re.compile(r'(-?\d+\.?\d*)\s*(?:/|÷|divided\s+by)\s*(-?\d+\.?\d*)'), 2),
            'power': (re.compile(r'(-?\d+\.?\d*)\s*(?:\*\*|to\s+the\s+power\s+of|raised\s+to)\s*(-?\d+\.?\d*)'), 2),
            'modulo': (re.compile(r'(-?\d+\.?\d*)\s*(?:%|mod(?:ulo)?)\s*(-?\d+\.?\d*)'), 2),
            'sqrt': (re.compile(r'(?:square\s+root\s+(?:of\s+)?|sqrt\s+(?:of\s+)?)(-?\d+\.?\d*)'), 1),
            'factorial': (re.compile(r'(?:factorial\s+(?:of\s+)?|(\d+)\s*(?:factorial|!))(\d*)'), 1),
            'fibonacci': (re.compile(r'(\d+)(?:th|st|nd|rd)?\s*(?:fibonacci|fib)'), 1),
            'prime_check': (re.compile(r'(?:is\s+)?(\d+)\s*(?:a\s+)?(?:prime|primality)'), 1),
            'sort_asc': (re.compile(r'sort(?:\s+ascending)?\s+(.+?)(?:\s+in\s+ascending)?$'), 1),
            'sort_desc': (re.compile(r'sort\s+(?:descending|from\s+(?:high|large))(.+)'), 1),
            'reverse': (re.compile(r'reverse\s+(?:the\s+)?(?:string\s+|word\s+|text\s+)?["\']?(\w+)["\']?'), 1),
            'length': (re.compile(r'(?:length\s+(?:of\s+)?|how\s+(?:long|many\s+(?:char|letter))\w*\s+(?:is\s+|in\s+)?)["\']?(\w+)["\']?'), 1),
            'abs_val': (re.compile(r'(?:abs(?:olute)?\s+(?:value\s+)?(?:of\s+)?)(-?\d+\.?\d*)'), 1),
            'convert_bin': (re.compile(r'(\d+)\s*(?:to|in)\s*binary'), 1),
            'convert_hex': (re.compile(r'(\d+)\s*(?:to|in)\s*hex'), 1),
        }

    def compute(self, query, trace=None):
        """Fused computation: query → W* → operation → execute → result.

        The basis determines WHAT operation to perform.
        Python executes it deterministically.
        The result flows back as language.
        """
        if trace is None:
            trace = []

        if self.W_op is None:
            return None, False

        # Step 1: Embed the query
        q_emb = self.embeddings.embed(query, self.tokenizer)
        if q_emb is None:
            return None, False

        # Step 2: W* determines the operation
        # op_scores = q_emb @ W_op → scores for each operation
        op_scores = q_emb.astype(np.float32) @ self.W_op  # (n_ops,)

        # Softmax to get probabilities
        op_scores = op_scores - op_scores.max()  # numerical stability
        op_probs = np.exp(op_scores) / np.sum(np.exp(op_scores))

        # Top operation
        best_idx = int(np.argmax(op_probs))
        best_op = self.op_names[best_idx]
        best_prob = float(op_probs[best_idx])

        if best_prob < 0.05:  # Not confident enough
            return None, False

        # Also check: does this query actually look computational?
        # Require at least one digit or computation keyword
        lower = query.lower()
        has_numbers = bool(re.search(r'\d', lower))
        has_compute_words = any(w in lower for w in [
            'calculate', 'compute', 'what is', 'how much', 'how many',
            'sort', 'reverse', 'convert', 'factorial', 'fibonacci',
            'prime', 'square root', 'binary', 'hex', 'length',
            'plus', 'minus', 'times', 'divided', 'add', 'subtract',
            'multiply', 'power', 'modulo', 'absolute',
        ])

        if not has_numbers and not has_compute_words:
            return None, False

        trace.append(f"[ComputeBasis] W* selected: {best_op} "
                     f"(prob={best_prob:.3f})")

        # Step 3: Extract arguments
        args = self._extract_args(best_op, query)
        if args is None:
            # Try second-best operation
            second_idx = int(np.argsort(-op_probs)[1])
            second_op = self.op_names[second_idx]
            args = self._extract_args(second_op, query)
            if args is not None:
                best_op = second_op
                trace.append(f"[ComputeBasis] Fallback to: {best_op}")
            else:
                return None, False

        # Step 4: Build and execute code from template
        template, n_args = self.op_templates[best_op]

        if n_args == 1:
            code = template.format(a=args[0])
        elif n_args == 2:
            code = template.format(a=args[0], b=args[1])
        else:
            return None, False

        trace.append(f"[ComputeBasis] Executing: {code[:80]}")

        # Step 5: Safe execution
        result = self._safe_execute(code, trace)
        if result is None:
            return None, False

        # Step 6: Format result
        response = self._format_result(best_op, args, result, query)
        trace.append(f"[ComputeBasis] Result: {response[:100]}")

        return response, True

    def _extract_args(self, op_name, query):
        """Extract arguments for an operation from the query."""
        if op_name not in self.op_extractors:
            return None

        pattern, n_args = self.op_extractors[op_name]
        lower = query.lower()

        match = pattern.search(lower)
        if not match:
            # Fallback: extract all numbers from the query
            nums = re.findall(r'-?\d+\.?\d*', query)
            if n_args == 1 and len(nums) >= 1:
                return [nums[0]]
            elif n_args == 2 and len(nums) >= 2:
                return [nums[0], nums[1]]
            # For sort/reverse, extract the content differently
            if op_name in ('sort_asc', 'sort_desc') and nums:
                return ['[' + ','.join(nums) + ']']
            if op_name == 'reverse':
                # Find a word to reverse
                rev_match = re.search(
                    r'reverse\s+(?:the\s+)?(?:string\s+|word\s+)?["\']?(\w+)',
                    lower)
                if rev_match:
                    return [f"'{rev_match.group(1)}'"]
            return None

        groups = [g for g in match.groups() if g is not None and g.strip()]
        if not groups:
            return None

        # For factorial: handle "5 factorial" vs "factorial of 5"
        if op_name == 'factorial':
            nums = re.findall(r'\d+', query)
            if nums:
                return [nums[0]]

        if n_args == 1:
            return [groups[0].strip()]
        elif n_args == 2 and len(groups) >= 2:
            return [groups[0].strip(), groups[1].strip()]

        return None

    def _safe_execute(self, code, trace):
        """Execute in sandboxed environment."""
        import math

        safe_builtins = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'len': len, 'sorted': sorted, 'reversed': reversed,
            'list': list, 'int': int, 'float': float, 'str': str,
            'range': range, 'all': all, 'any': any, 'bool': bool,
            'pow': pow, 'divmod': divmod, 'bin': bin, 'hex': hex, 'oct': oct,
            'True': True, 'False': False, 'None': None,
            'map': map, 'filter': filter,
        }

        # Reject dangerous code
        dangerous = ['import', 'exec', 'eval', 'open', 'file', '__',
                     'system', 'subprocess', 'os.', 'shutil', 'input',
                     'breakpoint', 'compile', 'globals', 'locals']
        code_lower = code.lower()
        # Allow __import__("math") specifically
        for d in dangerous:
            if d in code_lower and not (d == 'import' and 'math' in code_lower):
                trace.append(f"[ComputeBasis] Rejected: contains '{d}'")
                return None

        try:
            # Add math module access
            safe_globals = {"__builtins__": safe_builtins, "math": math}
            result = eval(code, safe_globals, {})
            return result
        except Exception as e:
            trace.append(f"[ComputeBasis] Error: {e}")
            return None

    def _format_result(self, op_name, args, result, query):
        """Format the computation result as natural language."""
        if op_name in ('add', 'subtract', 'multiply', 'divide', 'power', 'modulo'):
            ops = {'add': '+', 'subtract': '-', 'multiply': '×',
                   'divide': '÷', 'power': '^', 'modulo': '%'}
            op_sym = ops.get(op_name, '?')
            if isinstance(result, float) and result == int(result):
                result = int(result)
            return f"{args[0]} {op_sym} {args[1]} = {result}"

        elif op_name == 'sqrt':
            return f"√{args[0]} = {result}"

        elif op_name == 'factorial':
            return f"{args[0]}! = {result}"

        elif op_name == 'fibonacci':
            ordinal = args[0]
            return f"The {ordinal}th Fibonacci number is {result}."

        elif op_name == 'prime_check':
            n = args[0]
            if result:
                return f"Yes, {n} is a prime number."
            else:
                return f"No, {n} is not a prime number."

        elif op_name in ('sort_asc', 'sort_desc'):
            return str(result)

        elif op_name == 'reverse':
            return str(result).strip("'\"")

        elif op_name == 'length':
            return f"The length of {args[0]} is {result} characters."

        elif op_name == 'convert_bin':
            return f"{args[0]} in binary is {result}."

        elif op_name == 'convert_hex':
            return f"{args[0]} in hexadecimal is {result}."

        elif op_name == 'abs_val':
            return f"The absolute value of {args[0]} is {result}."

        else:
            return str(result)

    def get_confidence(self, query):
        """How confident is the compute basis that this query is computational?

        Returns (operation, probability) or (None, 0).
        """
        if self.W_op is None:
            return None, 0.0

        q_emb = self.embeddings.embed(query, self.tokenizer)
        if q_emb is None:
            return None, 0.0

        op_scores = q_emb.astype(np.float32) @ self.W_op
        op_scores = op_scores - op_scores.max()
        op_probs = np.exp(op_scores) / np.sum(np.exp(op_scores))

        best_idx = int(np.argmax(op_probs))
        return self.op_names[best_idx], float(op_probs[best_idx])
