"""
VEF Benchmark Suite
====================
Evaluate on public benchmarks. Downloads from HuggingFace automatically.

Usage:
    python benchmark.py                    # Run all benchmarks
    python benchmark.py --bench triviaqa   # Run specific benchmark
    python benchmark.py --limit 500        # Limit samples per benchmark

Benchmarks:
    triviaqa    - Open-domain factual Q&A (EM/F1)
    boolq       - Yes/No questions (accuracy)
    arc_easy    - Science reasoning, easy (accuracy)
    arc_challenge - Science reasoning, hard (accuracy)
    truthfulqa  - Truthfulness under pressure (MC accuracy)
    gsm8k       - Grade school math (exact match)
    mmlu        - 57-subject knowledge (accuracy)
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_dataset_hf(name, config=None, split='validation'):
    """Load from HuggingFace datasets."""
    from datasets import load_dataset
    if config:
        return load_dataset(name, config, split=split, trust_remote_code=True)
    return load_dataset(name, split=split, trust_remote_code=True)


def normalize(s):
    """Normalize text for comparison."""
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())


def pick_best_choice(model, query, choices):
    """Score by corpus compatibility + direct similarity.

    For each choice:
      1. Embed "question + choice" and find max corpus similarity
      2. Embed "choice" alone and compute similarity to embed("question")
    The choice most supported by the corpus wins.
    """
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    q_emb = model.embeddings.embed(query, model.tokenizer)
    if q_emb is None:
        return 0

    best_idx, best_score = 0, -1e9
    for j, choice in enumerate(choices):
        # Corpus compatibility: is Q+choice a natural pair?
        c_emb = model.embeddings.embed(f"{query} {choice}", model.tokenizer)
        corpus_score = 0.0
        if c_emb is not None:
            c_t = torch.tensor(c_emb, dtype=torch.float32, device=DEVICE)
            corpus_score = float((model.corpus.q_embeds @ c_t).max())

        # Direct: Q-choice semantic similarity
        ch_emb = model.embeddings.embed(choice, model.tokenizer)
        direct = float(q_emb @ ch_emb) if ch_emb is not None else 0.0

        score = corpus_score + direct * 0.5
        if score > best_score:
            best_score = score
            best_idx = j

    return best_idx


def f1_score(pred, gold):
    """Token-level F1."""
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 0.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred, golds):
    """Check if prediction matches any gold answer."""
    pred_n = normalize(pred)
    return any(pred_n == normalize(g) for g in golds)


# ═══════════════════════════════════════════════════════════
# BENCHMARKS
# ═══════════════════════════════════════════════════════════

def bench_triviaqa(model, limit=None):
    """TriviaQA — open-domain factual Q&A."""
    print("\n=== TriviaQA (closed-book) ===")
    ds = load_dataset_hf('mandarjoshi/trivia_qa', 'rc.nocontext', 'validation')
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"  Samples: {len(ds)}")

    correct, total, f1_sum = 0, 0, 0.0
    t0 = time.time()
    for i, ex in enumerate(ds):
        q = ex['question']
        golds = ex['answer']['aliases'] + [ex['answer']['value']]

        answer = model.reason(q)
        if exact_match(answer, golds):
            correct += 1
        best_f1 = max(f1_score(answer, g) for g in golds)
        f1_sum += best_f1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(ds)}] EM={100*correct/total:.1f}% "
                  f"F1={100*f1_sum/total:.1f}%")

    em = 100 * correct / max(total, 1)
    f1 = 100 * f1_sum / max(total, 1)
    elapsed = time.time() - t0
    print(f"  Result: EM={em:.1f}%, F1={f1:.1f}% ({elapsed:.0f}s)")
    return {'benchmark': 'triviaqa', 'em': em, 'f1': f1, 'n': total}


def bench_boolq(model, limit=None):
    """BoolQ — yes/no questions."""
    print("\n=== BoolQ ===")
    ds = load_dataset_hf('google/boolq', split='validation')
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"  Samples: {len(ds)}")

    correct, total = 0, 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        passage = ex['passage'][:300]
        q = ex['question']
        gold = ex['answer']  # True/False

        query = f"{passage}\n\nQuestion: {q}"
        answer = model.reason(query).lower().strip()

        pred = None
        if any(w in answer[:50] for w in ['yes', 'true', 'correct', 'right', 'affirmative']):
            pred = True
        elif any(w in answer[:50] for w in ['no', 'false', 'incorrect', 'wrong', "don't", "not", 'negative']):
            pred = False

        if pred is not None and pred == gold:
            correct += 1
        total += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(ds)}] Acc={100*correct/total:.1f}%")

    acc = 100 * correct / max(total, 1)
    elapsed = time.time() - t0
    print(f"  Result: Accuracy={acc:.1f}% ({elapsed:.0f}s)")
    return {'benchmark': 'boolq', 'accuracy': acc, 'n': total}


def bench_arc(model, difficulty='Challenge', limit=None):
    """ARC — science multiple choice via embedding similarity."""
    name = f"ARC-{difficulty}"
    print(f"\n=== {name} ===")
    ds = load_dataset_hf('allenai/ai2_arc', f'ARC-{difficulty}', 'test')
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"  Samples: {len(ds)}")

    correct, total = 0, 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        q = ex['question']
        labels = ex['choices']['label']
        texts = ex['choices']['text']
        gold = ex['answerKey']

        pred_idx = pick_best_choice(model, q, texts)
        if labels[pred_idx] == gold:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(ds)}] Acc={100*correct/total:.1f}%")

    acc = 100 * correct / max(total, 1)
    elapsed = time.time() - t0
    print(f"  Result: Accuracy={acc:.1f}% (random=25%) ({elapsed:.0f}s)")
    return {'benchmark': name.lower().replace('-', '_'), 'accuracy': acc, 'n': total}


def bench_truthfulqa(model, limit=None):
    """TruthfulQA — MC1 via embedding similarity."""
    print("\n=== TruthfulQA MC1 ===")
    ds = load_dataset_hf('truthfulqa/truthful_qa', 'multiple_choice',
                          split='validation')
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"  Samples: {len(ds)}")

    correct, total = 0, 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        q = ex['question']
        choices = ex['mc1_targets']['choices']
        labels = ex['mc1_targets']['labels']

        pred_idx = pick_best_choice(model, q, choices)
        if labels[pred_idx] == 1:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(ds)}] Acc={100*correct/total:.1f}%")

    acc = 100 * correct / max(total, 1)
    elapsed = time.time() - t0
    print(f"  Result: MC1 Accuracy={acc:.1f}% ({elapsed:.0f}s)")
    return {'benchmark': 'truthfulqa_mc1', 'accuracy': acc, 'n': total}


def bench_gsm8k(model, limit=None):
    """GSM8K — grade school math."""
    print("\n=== GSM8K ===")
    ds = load_dataset_hf('openai/gsm8k', 'main', 'test')
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"  Samples: {len(ds)}")

    correct, total = 0, 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        q = ex['question']
        gold_answer = ex['answer']
        # Extract final number from gold
        gold_num = re.findall(r'[\-\d,]+', gold_answer.split('####')[-1].strip())
        gold_num = gold_num[-1].replace(',', '') if gold_num else ''

        answer = model.reason(q)
        # Extract number from model answer
        pred_nums = re.findall(r'[\-\d,]+', answer)
        pred_num = pred_nums[-1].replace(',', '') if pred_nums else ''

        if pred_num == gold_num and gold_num:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(ds)}] Acc={100*correct/total:.1f}%")

    acc = 100 * correct / max(total, 1)
    elapsed = time.time() - t0
    print(f"  Result: Accuracy={acc:.1f}% ({elapsed:.0f}s)")
    return {'benchmark': 'gsm8k', 'accuracy': acc, 'n': total}


def bench_mmlu(model, limit=None):
    """MMLU — 57-subject knowledge test via embedding similarity."""
    print("\n=== MMLU ===")
    ds = load_dataset_hf('cais/mmlu', 'all', 'test')
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"  Samples: {len(ds)}")

    correct, total = 0, 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        q = ex['question']
        choices = ex['choices']
        gold_idx = ex['answer']

        pred_idx = pick_best_choice(model, q, choices)
        if pred_idx == gold_idx:
            correct += 1
        total += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(ds)}] Acc={100*correct/total:.1f}%")

    acc = 100 * correct / max(total, 1)
    elapsed = time.time() - t0
    print(f"  Result: Accuracy={acc:.1f}% (random=25%) ({elapsed:.0f}s)")
    return {'benchmark': 'mmlu', 'accuracy': acc, 'n': total}


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

BENCHMARKS = {
    'triviaqa': bench_triviaqa,
    'boolq': bench_boolq,
    'arc_easy': lambda m, l: bench_arc(m, 'Easy', l),
    'arc_challenge': lambda m, l: bench_arc(m, 'Challenge', l),
    'truthfulqa': bench_truthfulqa,
    'gsm8k': bench_gsm8k,
    'mmlu': bench_mmlu,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VEF Benchmark Suite')
    parser.add_argument('--bench', type=str, default=None,
                        help='Run specific benchmark (or "all")')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit samples per benchmark')
    args = parser.parse_args()

    from model import VEF
    print("Loading VEF model...")
    model = VEF()

    to_run = list(BENCHMARKS.keys()) if not args.bench or args.bench == 'all' else [args.bench]

    results = []
    for name in to_run:
        if name not in BENCHMARKS:
            print(f"Unknown benchmark: {name}")
            continue
        try:
            r = BENCHMARKS[name](model, args.limit)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Benchmark':<20} {'Metric':<10} {'Score':>8} {'N':>8}")
    print("-" * 50)
    for r in results:
        name = r['benchmark']
        if 'em' in r:
            print(f"{name:<20} {'EM':<10} {r['em']:>7.1f}% {r['n']:>8}")
            print(f"{'':<20} {'F1':<10} {r['f1']:>7.1f}%")
        else:
            print(f"{name:<20} {'Accuracy':<10} {r['accuracy']:>7.1f}% {r['n']:>8}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
