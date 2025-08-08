index_dir2=/tmp2/clueweb22/indexes/pyserini_index

fuse_path1="/tmp2/intern/guanwei/ikat-baseline/EXP/adarewriter13_llm4csnpr_cand10_splade/IKAT2025/run.json"
fuse_path2="/tmp2/intern/guanwei/ikat-baseline/EXP/adarewriter13_chiqnpr_cand10_splade/IKAT2025/run.json"
fuse_output="/tmp2/intern/guanwei/ikat-baseline/EXP/chiq_llmcs_fused_rrf/IKAT2025/adaR13_chiqnpr_llm4csnpr_run.json"
query_path="/tmp2/intern/guanwei/DATA/queries/queries_adarewriter13_llm4csnpr_cand10_2025.tsv"
rerank_output="/tmp2/intern/guanwei/ikat-baseline/EXP/chiq_llmcs_fused_rrf_rerank/IKAT2025/adaR13_chiqnpr_llm4csnpr_run.json"

# 1. Perform RRF fusion on two different runs
python RRF/fuse_rrf.py \
  --run_paths ${fuse_path1} ${fuse_path2} \
  --output ${fuse_output} \
  --k 60

# 2. Rerank the fused run using DeBERTa-v3 model
CUDA_VISIBLE_DEVICES=1 python -m rerank.rerank \
                        --index_path $index_dir2 \
                        --model naver/trecdl22-crossencoder-debertav3 \
                        --run ${fuse_output} \
                        --query_file ${query_path} \
                        --output ${rerank_output}
