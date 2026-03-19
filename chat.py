"""
VEF Chat — Interactive REPL
=============================
Zero gradient descent. Fully interpretable. Every answer traced.

    python chat.py
    python chat.py --no-reasoning
    python chat.py --data path/to/data
    python chat.py --system "You are a pirate assistant."
"""
import sys
import os
import time
import textwrap
import argparse

parser = argparse.ArgumentParser(description="VEF Chat", add_help=False)
parser.add_argument('--data', type=str, default=None,
                    help='Path to data/ directory')
parser.add_argument('--no-reasoning', action='store_true',
                    help='Hide reasoning trace')
parser.add_argument('--system', type=str, default=None,
                    help='Custom system prompt')
parser.add_argument('-h', '--help', action='store_true')
args, _ = parser.parse_known_args()

if args.data:
    import model as m
    m.DATA_DIR = args.data

from model import VEF

# Terminal colors
R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[90m"
G   = "\033[32m"
CYN = "\033[36m"

# Response wrapping width
WRAP = 80


def wrap(text, indent=""):
    """Word-wrap text for terminal readability."""
    lines = text.split('\n')
    wrapped = []
    for line in lines:
        if len(line) > WRAP:
            wrapped.extend(textwrap.fill(line, width=WRAP,
                           subsequent_indent=indent).split('\n'))
        else:
            wrapped.append(line)
    return '\n'.join(wrapped)


def main():
    if args.help:
        print(f"""
{B}VEF Chat — Zero Gradient Descent Language Model{R}

Usage: python chat.py [options]

Options:
  --no-reasoning    Hide the reasoning trace
  --data DIR        Use a custom data directory
  --system "..."    Set a custom system prompt

Commands (during chat):
  /reasoning        Toggle reasoning trace on/off
  /system <prompt>  Change the system prompt
  /clear            Clear the screen
  /help             Show this help
  exit              Quit
""")
        return

    print(f"\n{DIM}  Loading...{R}", end='', flush=True)
    vef = VEF(quiet=True)
    print(f"\r            \r", end='')
    show = not args.no_reasoning

    system_prompt = args.system or (
        'My name is VEF. I am a helpful language model built entirely from '
        'frozen statistics with zero gradient descent. I explain my reasoning '
        'and I am honest about what I do not know.')
    vef.set_system(system_prompt, show_reasoning=show)

    print(f"""
{B}  VEF{R}
{DIM}  {vef.corpus.n_entries:,} entries | {vef.embeddings.vocab_size:,} vocab | {vef.embeddings.dim}d | {vef.load_time:.1f}s
  /help for commands | /reasoning to toggle trace{R}
""")

    while True:
        try:
            user = input(f"{B}You: {R}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{R}")
            break

        if not user:
            continue

        # Commands
        lower = user.lower()
        if lower in ('exit', 'quit', 'bye', '/exit', '/quit'):
            print(f"{DIM}Goodbye!{R}")
            break

        if lower in ('/reasoning', '/trace'):
            show = not show
            vef.set_system(system_prompt, show_reasoning=show)
            print(f"{DIM}  Trace: {'on' if show else 'off'}{R}\n")
            continue

        if lower.startswith('/system'):
            new_prompt = user[7:].strip()
            if new_prompt:
                system_prompt = new_prompt
                vef.set_system(system_prompt, show_reasoning=show)
                print(f"{DIM}  System prompt updated.{R}\n")
            else:
                print(f"{DIM}  Current: {system_prompt[:80]}...{R}\n")
            continue

        if lower == '/clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            continue

        if lower == '/help':
            print(f"""{DIM}  Commands:
    /reasoning   Toggle reasoning trace
    /system ...  Change system prompt
    /clear       Clear screen
    /help        Show this help
    exit         Quit{R}
""")
            continue

        # Answer
        t0 = time.time()
        answer = vef.reason(user)
        dt = time.time() - t0

        if '\n\n[Reasoning]' in answer:
            main_answer, reasoning = answer.split('\n\n[Reasoning]', 1)
            print(f"\n{G}{B}VEF:{R} {wrap(main_answer, '     ')}")
            print(f"\n{DIM}[Reasoning]{reasoning}{R}")
        else:
            print(f"\n{G}{B}VEF:{R} {wrap(answer, '     ')}")

        print(f"{DIM}  [{dt:.2f}s]{R}\n")


if __name__ == '__main__':
    main()
