# baseline.py
import random
import time
import json
import RNA
from eval_harness import evaluate


def make_hairpin(stem, loop):
    """Target hairpin: stem '(' + loop '.' + stem ')'."""
    return "(" * stem + "." * loop + ")" * stem


def make_targets(n=30, seed=0):
    """Sample hairpin targets with stem 3-8 and loop 3-6."""
    rng = random.Random(seed)
    targets = []
    for _ in range(n):
        stem = rng.randint(3, 8)
        loop = rng.randint(3, 6)
        targets.append(make_hairpin(stem, loop))
    return targets


def random_start(length, rng):
    return "".join(rng.choice("ACGU") for _ in range(length))


def run_rnainverse(targets, seed=0):
    """Design one sequence per target with ViennaRNA's RNAinverse."""
    rng = random.Random(seed)
    designs, times = [], []
    for target in targets:
        start = random_start(len(target), rng)
        t0 = time.perf_counter()
        seq, _dist = RNA.inverse_fold(start, target)
        times.append(time.perf_counter() - t0)
        designs.append(seq)
    return designs, times


if __name__ == "__main__":
    targets = make_targets(n=30, seed=0)
    designs, times = run_rnainverse(targets, seed=0)

    res = evaluate(designs, targets)
    res["mean_time_s"] = sum(times) / len(times)

    print(f"n = {res['n']}")
    print(f"success rate = {res['success_rate']:.2f}")
    print(f"mean ensemble defect = {res['mean_ensemble_defect']:.3f}")
    print(f"mean time per target = {res['mean_time_s']:.3f} s")

    # Save raw results so later comparisons use the same numbers
    with open("baseline_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("Saved baseline_results.json")