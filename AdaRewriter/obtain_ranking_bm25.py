import argparse
import json

import pytrec_eval
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

# from shared_utils import get_logger
# logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="qrecc", required=True, choices=["qrecc", "ikat"], help="Dataset type, choose from 'qrecc' or 'ikat'.")
    parser.add_argument('--pyserini_index_path', type=str, required=True, help="Path to the Pyserini index directory.")
    parser.add_argument('--qrel_path', type=str, required=True, help="Path to qrel file for evaluation.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to input test jsonl file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the retrieved results.")
    parser.add_argument('--top_k', type=int, default=100, help="Number of passages to retrieve per query.")
    parser.add_argument('--cand_num', type=int, required=True, help="Number of rewrite candidates per query.")
    args = parser.parse_args()

    if args.task == "qrecc":
        (k_1, b) = (0.82, 0.68)
    else:
        (k_1, b) = (0.9, 0.4)

    searcher = LuceneSearcher(args.pyserini_index_path)
    searcher.set_bm25(k_1, b)

    qrels = json.load(open(args.qrel_path))
    input_f = open(args.data_path)
    output_f = open(args.output_path, "w")
    num = len(open(args.data_path).readlines())

    for line in tqdm(input_f, total=num):
        data = json.loads(line)
        if data['sample_id'] not in qrels:
            continue
        
        query_candidates = []
        query_ids = []
        cur_session_qrels = {}

        for i in range(args.cand_num):
            query_candidates.append(f"{data['predicted_rewrite'][i]} {data['predicted_response'][i]}")
            query_ids.append(f"{data['sample_id']}_{i}")
            cur_session_qrels[f"{data['sample_id']}_{i}"] = qrels[data['sample_id']]

        hits = searcher.batch_search(query_candidates, query_ids, args.top_k, threads=16)
        bm25_result = {}

        for qid in query_ids:
            for i, item in enumerate(hits[qid]):
                if qid not in bm25_result:
                    bm25_result[qid] = {}
                bm25_result[qid][item.docid] = - i - 1 + args.top_k

        evaluator = pytrec_eval.RelevanceEvaluator(cur_session_qrels, {"recip_rank"})
        eval_metrics = evaluator.evaluate(bm25_result)

        cur_session_scores = []

        for i in range(args.cand_num):
            cur_session_scores.append(eval_metrics[f"{data['sample_id']}_{i}"]['recip_rank'])

        indexed_scores = []
        for i, score in enumerate(cur_session_scores):
            rewrite = data['predicted_rewrite'][i]
            response = data['predicted_response'][i]
            total_len = len((rewrite + response).strip())
            indexed_scores.append((i, score, total_len))

        indexed_scores.sort(key=lambda x: (-x[1], -x[2]))

        ranks = [0] * args.cand_num
        for rank, (idx, _, _) in enumerate(indexed_scores):
            ranks[idx] = rank

        out_data = data
        out_data['bm25_score'] = cur_session_scores
        out_data['ranks'] = ranks

        output_f.write(json.dumps(out_data) + "\n")
        output_f.flush()

    input_f.close()
    output_f.close()
    
if __name__ == "__main__":
    main()