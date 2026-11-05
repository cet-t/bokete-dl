import argparse
import sys
from . import bulk, data, generate, generate_csv, review
from .llm import LLMError


def main():
    parser = argparse.ArgumentParser(prog="bokete-dl")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--revision", required=True); p.add_argument("--out", required=True); p.add_argument("--threshold", type=float, default=.7)
    p = sub.add_parser("split")
    p.add_argument("--input", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--model", required=True); p.add_argument("--mode", choices=("examples", "no-examples"), required=True); p.add_argument("--theme", default=""); p.add_argument("--count", type=int, required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--splits", required=True); p.add_argument("--out", required=True); p.add_argument("--threshold", type=float, default=.7)
    p = sub.add_parser("generate-themes", help="???????????")
    p.add_argument("--model", required=True); p.add_argument("--mode", choices=("examples", "no-examples"), required=True); p.add_argument("--themes", nargs="+", required=True); p.add_argument("--count", type=int, default=100); p.add_argument("--seed", type=int, required=True); p.add_argument("--splits", required=True); p.add_argument("--out", required=True); p.add_argument("--threshold", type=float, default=.7)
    p = sub.add_parser("generate-csv", help="???????????? CSV ???")
    p.add_argument("--model", required=True); p.add_argument("--mode", choices=("examples", "no-examples"), required=True); p.add_argument("--themes", nargs="+", required=True); p.add_argument("--count", type=int, default=100); p.add_argument("--seed", type=int, required=True); p.add_argument("--splits", required=True); p.add_argument("--out", required=True); p.add_argument("--threshold", type=float, default=.7)
    p = sub.add_parser("answer")
    p.add_argument("--model", required=True); p.add_argument("--input", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("review")
    rs = p.add_subparsers(dest="action", required=True)
    p = rs.add_parser("export")
    p.add_argument("--input", required=True, nargs="+"); p.add_argument("--seed", type=int, required=True); p.add_argument("--out", required=True); p.add_argument("--reviewers", type=int, default=2); p.add_argument("--answers"); p.add_argument("--show-answers", action="store_true")
    p = rs.add_parser("summarize")
    p.add_argument("--input", required=True); p.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare": data.prepare(args.revision, args.out, args.threshold)
        elif args.command == "split": data.split(args.input, args.seed, args.out)
        elif args.command == "generate": generate.generate(args.model, args.mode, args.theme, args.count, args.seed, args.splits, args.out, args.threshold)
        elif args.command == "generate-themes": bulk.generate_themes(args.model, args.mode, args.themes, args.count, args.seed, args.splits, args.out, args.threshold)
        elif args.command == "generate-csv": generate_csv.generate_csv(args.model, args.mode, args.themes, args.count, args.seed, args.splits, args.out, args.threshold)
        elif args.command == "answer": generate.answer(args.model, args.input, args.seed, args.out)
        elif args.action == "export": review.export(args.input, args.seed, args.out, args.reviewers, args.answers, args.show_answers)
        else: review.summarize(args.input, args.out)
    except (ValueError, OSError, KeyError, LLMError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
