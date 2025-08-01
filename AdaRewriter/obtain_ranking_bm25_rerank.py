import argparse
import json
import torch
import pytrec_eval
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from torch.cuda.amp import autocast

def main():
    parser = argparse.ArgumentParser(description="Generate preference data for reward model training using a BM25+Reranker pipeline.")
    parser.add_argument('--task', type=str, default="qrecc", help="Task name, e.g., 'qrecc'.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input data file with predicted rewrites.")
    parser.add_argument('--pyserini_index_path', type=str, required=True, help="Path to the Pyserini Lucene index.")
    parser.add_argument('--qrel_path', type=str, required=True, help="Path to the ground truth qrels file.")
    parser.add_argument('--output_path', type=str, default="./preference_data_reranked.jsonl", help="Path for the output preference data.")
    parser.add_argument('--top_k', type=int, default=100, help="Number of documents to retrieve in the initial BM25 stage.")
    parser.add_argument('--cand_num', type=int, default=32, help="Number of candidate queries to process per sample.")
    parser.add_argument('--reranker_model', type=str, default="naver/trecdl22-crossencoder-debertav3", help="Name or path of the cross-encoder model for reranking.")
    args = parser.parse_args()

    if args.task == "qrecc":
        (k_1, b) = (0.82, 0.68)
    else:
        (k_1, b) = (0.9, 0.4)
        
    searcher = LuceneSearcher(args.pyserini_index_path)
    searcher.set_bm25(k_1, b)

    reranker = CrossEncoder(args.reranker_model, device=device)
    print(f"[INFO] Models initialized. Reranker running on {device}.")

    qrels = json.load(args.qrel_path)
    if args.task == "qrecc":
        qrels = dict(filter(lambda x: x[1] != {"": 1}, qrels.items()))

    input_f = open(args.data_path)
    output_f = open(args.output_path, "w")
    num_lines = len(open(args.data_path).readlines())

    for line in tqdm(input_f, total=num_lines, desc="Generating Preferences"):
        data = json.loads(line)
        sample_id = data['sample_id']

        if sample_id not in qrels:
            continue
        
        query_candidates = [f"{data['predicted_rewrite'][i]} {data['predicted_response'][i]}" for i in range(args.cand_num)]
        query_ids = [f"{sample_id}_{i}" for i in range(args.cand_num)]

        # Step 1: Initial BM25 Retrieval
        bm25_hits = searcher.batch_search(query_candidates, query_ids, args.top_k, threads=16)

        reranked_results_for_eval = {}
        
        # Step 2: Rerank results for each candidate
        for i, qid in enumerate(query_ids):
            current_query_text = query_candidates[i]
            doc_ids = [hit.docid for hit in bm25_hits.get(qid, [])]
            if not doc_ids:
                reranked_results_for_eval[qid] = {}
                continue

            doc_contents_map = searcher.batch_doc(doc_ids, threads=16)
            
            reranker_pairs, valid_doc_ids = [], []
            for doc_id in doc_ids:
                raw_doc = doc_contents_map.get(doc_id)
                if raw_doc:
                    content = json.loads(raw_doc.raw()).get('contents', '')
                    reranker_pairs.append([current_query_text, content])
                    valid_doc_ids.append(doc_id)
            
            if not reranker_pairs:
                reranked_results_for_eval[qid] = {}
                continue

            with autocast():
                reranker_scores = reranker.predict(reranker_pairs, batch_size=32, show_progress_bar=False)
            sorted_reranked = sorted(zip(valid_doc_ids, reranker_scores), key=lambda x: x[1], reverse=True)
            reranked_results_for_eval[qid] = {doc_id: float(score) for doc_id, score in sorted_reranked}

        # Step 3: Evaluate the RERANKED results
        cur_session_qrels = {qid: qrels[sample_id] for qid in query_ids}
        evaluator = pytrec_eval.RelevanceEvaluator(cur_session_qrels, {"ndcg_cut.10"})
        eval_metrics = evaluator.evaluate(reranked_results_for_eval)

        # Step 4: Create Preference Ranks based on nDCG@10
        indexed_scores = []
        for i, qid in enumerate(query_ids):
            score = eval_metrics.get(qid, {}).get('ndcg_cut_10', 0.0)
            total_len = len(query_candidates[i].strip())
            indexed_scores.append((i, score, total_len))

        indexed_scores.sort(key=lambda x: (-x[1], -x[2]))

        ranks = [0] * args.cand_num
        for rank, (original_index, _, _) in enumerate(indexed_scores):
            ranks[original_index] = rank

        out_data = data
        final_scores = [item[1] for item in sorted(indexed_scores, key=lambda x: x[0])]
        
        out_data['final_score'] = final_scores
        out_data['ranks'] = ranks
        
        output_f.write(json.dumps(out_data) + "\n")
        output_f.flush()
    
    input_f.close()
    output_f.close()
    print(f"\n[INFO] Aligned preference data saved to {args.output_path}")

if __name__ == "__main__":
    main()