"""CLI: synthesize RAFT training data from the vault.

  python -m rhizome.synth --vault vault --out data --generator stub
  python -m rhizome.synth --vault vault --out data --generator claude   # needs subscription token
"""
import argparse

from .pipeline import run_synth
from .generator import StubGenerator, ClaudeMCPGenerator


def main():
    ap = argparse.ArgumentParser(prog="rhizome.synth",
                                 description="Synthesize RAFT training data from a vault.")
    ap.add_argument("--vault", default="vault")
    ap.add_argument("--out", default="data")
    ap.add_argument("--generator", choices=["stub", "claude"], default="stub")
    ap.add_argument("--n-questions", type=int, default=3)
    ap.add_argument("--k-distractors", type=int, default=4)
    ap.add_argument("--p-golden", type=float, default=0.8)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--cross-top", type=int, default=1,
                    help="research chunks paired per note for cross layer examples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gen = (StubGenerator(args.n_questions) if args.generator == "stub"
           else ClaudeMCPGenerator(args.n_questions))

    stats = run_synth(args.vault, args.out, gen,
                      k_distractors=args.k_distractors, p_golden=args.p_golden,
                      eval_frac=args.eval_frac, cross_top=args.cross_top, seed=args.seed)
    print("RAFT synth complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
