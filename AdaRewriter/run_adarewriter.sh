# ====================================
# ========== Training Phase ==========
# ====================================

# 2. Ranking Assessment
index_path="/tmp2/QReCC/indexes/pyserini_index"

qrel_path="/tmp2/intern/guanwei/DATA/topics/qrecc_train_qrels_binary.json"
input_path="/tmp2/intern/guanwei/DATA/candidates/qrecc_train_llama_temp07_cand05_all.jsonl"
output_path="/tmp2/intern/guanwei/DATA/candidates/qrecc_train_llama_temp07_cand05_all_wranks.jsonl"

N_candidates=5

python obtain_ranking_bm25.py \
--task="qrecc" \
--top_k=500 \
--qrel_path=$qrel_path \
--output_path=$output_path \
--pyserini_index_path=$index_path \
--data_path=$input_path \
--cand_num=$N_candidates

# 3. Training AdaRewriter
checkpoint="/tmp2/intern/guanwei/AdaRewriter/checkpoints/qrecc_epoch13_lr5r-6.ckpt"

input_path="/tmp2/intern/guanwei/DATA/candidates/qrecc_train_llama_temp07_cand05_all_wranks.jsonl"
num_epochs=13
learning_rate=5e-6
N_candidates=5

CUDA_VISIBLE_DEVICES=2 python main.py \
--default_root_dir="./checkpoints" \
--train_path=$input_path \
--cand_num=$N_candidates \
--train_batch_size=1 \
--valid_batch_size=1 \
--devices=1 \
--gradient_accumulate=4 \
--loss_margin=0.1 \
--max_epochs=$num_epochs \
--lr=$learning_rate \
--loss_type="weight-divide" \
--add_context \
# --checkpoint=$checkpoint

# ====================================
# ========== Testing Phase ===========
# ====================================

# 4. Testing AdaRewriter
checkpoint="/tmp2/intern/guanwei/AdaRewriter/checkpoints/qrecc_epoch13_lr5r-6.ckpt"

input_path="/tmp2/intern/guanwei/DATA/candidates/ikat2025_test_llm4csnpr_temp07_cand10.jsonl"
output_file="/tmp2/intern/guanwei/DATA/queries/queries_adarewriter13_llm4csnpr_cand10_2025.tsv"

N_candidates=10

CUDA_VISIBLE_DEVICES=2 python main.py \
--test_path=$input_path \
--cand_num=$N_candidates \
--test \
--output_file=$output_file \
--add_context \
--checkpoint=$checkpoint