"""Generate a small synthetic 'TinyStories-like' corpus so the whole pipeline can be tested offline.

    python scripts/make_toy_corpus.py --out data/toy --n-train 4000 --n-valid 300
"""
import argparse
import os
import random

NAMES = ["Lily", "Tom", "Anna", "Ben", "Mia", "Sam", "Zoe", "Max", "Ella", "Leo"]
ANIMALS = ["cat", "dog", "bird", "rabbit", "fish", "bear", "duck", "frog"]
THINGS = ["ball", "kite", "cookie", "toy car", "red hat", "big book", "shiny stone", "little boat"]
PLACES = ["park", "garden", "forest", "beach", "farm", "kitchen", "small town"]
FEELINGS = ["happy", "sad", "excited", "scared", "proud", "sleepy"]
ADJ = ["little", "big", "kind", "funny", "brave", "shy"]
ACTIONS = ["ran", "jumped", "laughed", "sang", "danced", "waited", "looked around"]


def story(rng: random.Random) -> str:
    n, m = rng.choice(NAMES), rng.choice([x for x in NAMES])
    a, t, p = rng.choice(ANIMALS), rng.choice(THINGS), rng.choice(PLACES)
    f1, f2, adj = rng.choice(FEELINGS), rng.choice(FEELINGS), rng.choice(ADJ)
    act = rng.choice(ACTIONS)
    lines = [
        f"Once upon a time, there was a {adj} girl named {n}. She loved to play at the {p}.",
        f"One day, {n} found a {t}. She felt {f1}. \"Look at this!\" she said to her friend {m}.",
        f"Then a {a} came and {act}. {m} said, \"Can we play with the {a}?\" {n} smiled and said yes.",
        f"They played all day. In the end, {n} was {f2}, and the {a} was her friend. The end.",
    ]
    k = rng.randint(3, 4)
    return " ".join(lines[:k])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-valid", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    for name, n in (("train", args.n_train), ("valid", args.n_valid)):
        with open(os.path.join(args.out, f"{name}.txt"), "w", encoding="utf-8") as f:
            for _ in range(n):
                f.write(story(rng) + "\n<|endoftext|>\n")
    print(f"wrote toy corpus to {args.out}")


if __name__ == "__main__":
    main()
