import json
import argparse
from collections import defaultdict
import os

def load_run(path):
    with open(path) as f:
        return json.load(f)

def fuse_rrf(run_dicts, k=60):
    fused = defaultdict(dict)
    for run in run_dicts:
        for qid, doc_scores in run.items():
            # 排名 (sorted by descending score)
            ranked_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
            for rank, (docid, _) in enumerate(ranked_docs):
                fused[qid][docid] = fused[qid].get(docid, 0) + 1.0 / (k + rank + 1)
    return fused

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_paths", nargs="+", required=True, help="Paths to run.json files")
    parser.add_argument("--output", required=True, help="Path to save fused run.json")
    parser.add_argument("--k", type=int, default=60, help="RRF constant k")
    args = parser.parse_args()

    runs = [load_run(p) for p in args.run_paths]
    fused = fuse_rrf(runs, k=args.k)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(fused, f, indent=2)
        print(f"Fused run saved to {args.output}")

if __name__ == "__main__":
    main()