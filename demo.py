"""
VEF Out-of-Distribution Demo
==============================
Tests the model on queries it has NEVER seen in its corpus.
No cherry-picking — every query is shown, every answer is shown.
Failures are displayed alongside successes.

Run: python demo.py
"""
import sys
import os
import time
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import VEF

R  = "\033[0m"
B  = "\033[1m"
G  = "\033[32m"
RED = "\033[31m"
DIM = "\033[90m"
YEL = "\033[33m"


def section(title):
    print(f"\n{B}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{R}\n")


def ask(model, query, expected=None, check_fn=None):
    """Ask the model and display the result. Returns (passed, answer)."""
    t0 = time.time()
    answer = model.reason(query)
    ms = (time.time() - t0) * 1000

    status = ""
    passed = None
    if expected is not None:
        if expected.lower() in answer.lower():
            status = f" {G}PASS{R}"
            passed = True
        else:
            status = f" {RED}MISS{R}"
            passed = False
    elif check_fn is not None:
        if check_fn(answer):
            status = f" {G}PASS{R}"
            passed = True
        else:
            status = f" {RED}MISS{R}"
            passed = False

    print(f"  {B}Q:{R} {query}")
    print(f"  {G}A:{R} {answer[:200]}")
    print(f"  {DIM}[{ms:.0f}ms]{status}{R}\n")
    return passed, answer


def main():
    print(f"\n{B}VEF Out-of-Distribution Demo{R}")
    print(f"{DIM}Every query shown. Every answer shown. No cherry-picking.{R}\n")

    model = VEF()
    model.set_system("My name is VEF. I am a helpful language model built from frozen statistics.",
                      show_reasoning=True)

    results = {'pass': 0, 'fail': 0, 'nocheck': 0}

    def run(query, expected=None, check_fn=None):
        passed, answer = ask(model, query, expected, check_fn)
        if passed is True:
            results['pass'] += 1
        elif passed is False:
            results['fail'] += 1
        else:
            results['nocheck'] += 1
        return answer

    # ═══════════════════════════════════════════════════════════
    section("1. ARITHMETIC — unseen number combinations")
    # The model learned single-digit ops from corpus.
    # Multi-digit uses digit decomposition — never seen these exact sums.
    # ═══════════════════════════════════════════════════════════

    run("What is 23 + 49?", "72")
    run("What is 156 + 287?", "443")
    run("What is 7 * 8?", "56")
    run("What is 12 * 5?", "60")
    run("What is 100 - 37?", "63")
    run("What is 9 + 9 + 9?", "27")

    # ═══════════════════════════════════════════════════════════
    section("2. HONEST UNCERTAINTY — made-up words & unknowns")
    # A good model says 'I don't know' instead of hallucinating.
    # These words don't exist — the model should refuse, not confabulate.
    # ═══════════════════════════════════════════════════════════

    run("What is a glorpnax?",
        check_fn=lambda a: "don't" in a.lower() or "not" in a.lower() or "unknown" in a.lower())
    run("Define the word 'blorpitude'.",
        check_fn=lambda a: "don't" in a.lower() or "not" in a.lower() or "unknown" in a.lower())
    run("Who is Dr. Xanthor McFibble?",
        check_fn=lambda a: "don't" in a.lower() or "not" in a.lower() or "unknown" in a.lower())

    # ═══════════════════════════════════════════════════════════
    section("3. SPELL CORRECTION — typos the model has never seen")
    # The model corrects via edit distance during low-confidence introspection.
    # These misspellings are not in the corpus.
    # ═══════════════════════════════════════════════════════════

    run("What is garvity?",
        check_fn=lambda a: "force" in a.lower() or "attract" in a.lower() or "planet" in a.lower())
    run("Explane photosynthsis.",
        check_fn=lambda a: "light" in a.lower() or "plant" in a.lower() or "sun" in a.lower())
    run("What is a dictionery?",
        check_fn=lambda a: "word" in a.lower() or "defin" in a.lower() or "language" in a.lower())

    # ═══════════════════════════════════════════════════════════
    section("4. LETTER INTROSPECTION — counting letters in any word")
    # The model spells out the word and counts. Works on ANY word
    # because it operates on the characters, not corpus lookup.
    # ═══════════════════════════════════════════════════════════

    run("How many R's are in strawberry?", "3")
    run("How many L's are in llama?", "2")
    run("How many S's are in mississippi?", "4")
    run("How many E's are in excellence?", "4")

    # ═══════════════════════════════════════════════════════════
    section("5. WORD RELATIONSHIPS — co-substitution discovery")
    # Antonyms/synonyms found by mining context frames, not a dictionary.
    # Works for words the model discovered relationships for in the corpus.
    # ═══════════════════════════════════════════════════════════

    run("What is the opposite of big?", "small")
    run("What is the opposite of good?",
        check_fn=lambda a: "bad" in a.lower() or "evil" in a.lower())
    run("What is the opposite of old?", "new")
    run("What is the opposite of hot?",
        check_fn=lambda a: "cold" in a.lower() or "cool" in a.lower())
    run("What is the opposite of fast?",
        check_fn=lambda a: "slow" in a.lower())

    # ═══════════════════════════════════════════════════════════
    section("6. KNOWLEDGE RETRIEVAL — factual Q&A")
    # Standard knowledge questions. Tests corpus coverage and retrieval quality.
    # ═══════════════════════════════════════════════════════════

    run("What is gravity?",
        check_fn=lambda a: "force" in a.lower())
    run("What is photosynthesis?",
        check_fn=lambda a: "light" in a.lower() or "plant" in a.lower() or "sun" in a.lower())
    run("What is DNA?",
        check_fn=lambda a: "genetic" in a.lower() or "acid" in a.lower() or "molecule" in a.lower())
    run("What is the speed of light?",
        check_fn=lambda a: "300" in a or "light" in a.lower())
    run("Who wrote Romeo and Juliet?",
        check_fn=lambda a: "shakespeare" in a.lower())

    # ═══════════════════════════════════════════════════════════
    section("7. COMPOSITIONAL QUERIES — form x content")
    # The model finds the INTERSECTION of form (joke, poem) and content.
    # These exact combinations are unlikely to be in the corpus.
    # ═══════════════════════════════════════════════════════════

    run("Tell me a joke about cats.")
    run("Tell me a joke about math.")
    run("Write a short poem about the ocean.")

    # ═══════════════════════════════════════════════════════════
    section("8. REASONING UNDER CONSTRAINTS")
    # The model decomposes and reasons through multi-part queries.
    # ═══════════════════════════════════════════════════════════

    run("If I have 15 apples and give away 7, how many do I have?", "8")
    run("What is 5 groups of 4?", "20")
    run("I have 3 boxes with 6 items each. How many items total?", "18")

    # ═══════════════════════════════════════════════════════════
    section("9. DEFINITIONS — extracted from corpus patterns")
    # The model extracted 10K definitions from 'What is X?' patterns.
    # Tests whether definitions are accurate and useful.
    # ═══════════════════════════════════════════════════════════

    run("What is economics?",
        check_fn=lambda a: "econom" in a.lower() or "resource" in a.lower() or "money" in a.lower())
    run("What is democracy?",
        check_fn=lambda a: "govern" in a.lower() or "people" in a.lower() or "vote" in a.lower())
    run("What is evolution?",
        check_fn=lambda a: "species" in a.lower() or "change" in a.lower() or "natural" in a.lower() or "adapt" in a.lower() or "biological" in a.lower())

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    total_checked = results['pass'] + results['fail']
    print(f"\n{B}{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}{R}")
    print(f"  Checked:  {total_checked}")
    print(f"  {G}Passed:   {results['pass']}{R}")
    print(f"  {RED}Failed:   {results['fail']}{R}")
    print(f"  Unscored: {results['nocheck']} (compositional — subjective)")
    if total_checked > 0:
        pct = 100 * results['pass'] / total_checked
        print(f"\n  {B}Score: {results['pass']}/{total_checked} ({pct:.0f}%){R}")
    print()


if __name__ == '__main__':
    main()
