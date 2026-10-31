# eval_harness.py
import RNA
import numpy as np


def fold(seq):
    """Return (mfe_structure, mfe_energy) for a sequence."""
    structure, energy = RNA.fold(seq)
    return structure, energy


def is_success(seq, target):
    """True if the MFE structure exactly matches the target."""
    mfe_structure, _ = fold(seq)
    return mfe_structure == target


def ensemble_defect(seq, target):
    """Normalized ensemble defect of seq with respect to target (0 = perfect)."""
    fc = RNA.fold_compound(seq)
    fc.mfe()   # required before partition function
    fc.pf()
    return fc.ensemble_defect(target)


def evaluate(designs, targets):
    """
    designs: list of designed sequences
    targets: list of target structures (same length and order as designs)
    Returns summary metrics plus per-target results.
    """
    assert len(designs) == len(targets), "designs and targets must align"

    per_target = []
    for seq, target in zip(designs, targets):
        success = is_success(seq, target)
        defect = ensemble_defect(seq, target)
        per_target.append({
            "seq": seq,
            "target": target,
            "success": success,
            "ensemble_defect": defect,
            "length": len(seq),
        })

    success_rate = np.mean([r["success"] for r in per_target])
    mean_defect = np.mean([r["ensemble_defect"] for r in per_target])

    return {
        "n": len(per_target),
        "success_rate": float(success_rate),
        "mean_ensemble_defect": float(mean_defect),
        "per_target": per_target,
    }


# ---- Toy tests: run this file directly to check ----
if __name__ == "__main__":
    target = "((((...))))"

    # 1. Designs that should fold correctly
    good = "GGGGAAACCCC"
    assert is_success(good, target), "good design should succeed"

    # 2. Design that should fail (all G's, no pairing possible with C's)
    bad = "GGGGGGGGGGG"
    assert not is_success(bad, target), "bad design should fail"

    # 3. Defect sanity: a perfect fold should have a low defect,
    #    and the bad design should have a higher one
    d_good = ensemble_defect(good, target)
    d_bad = ensemble_defect(bad, target)
    assert d_good < d_bad, f"expected good < bad, got {d_good} vs {d_bad}"

    # 4. Batch evaluation
    res = evaluate([good, bad], [target, target])
    assert res["n"] == 2
    assert res["success_rate"] == 0.5

    print("All toy tests passed.")
    print(f"ensemble defect good={d_good:.3f}, bad={d_bad:.3f}")
    print(f"success rate on [good, bad]: {res['success_rate']}")