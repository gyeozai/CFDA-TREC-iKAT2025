# 1. Candidate Generation (gpt-4o-mini)
export OPENAI_API_KEY="your_openai_api_key"

input_path="/tmp2/intern/guanwei/DATA/topics/2025_flatten_test_topics.json"
output_path="/tmp2/intern/guanwei/DATA/candidates/ikat2025_test_llm4csnpr_temp10_cand10.jsonl"

N_candidates=10

python chat_prompt_RAR_CoT_v1.py \
--test_file_path=$input_path \
--demo_file_path="./demonstrations.json" \
--output_path=$output_path \
--n_generation=$N_candidates \
--omit_pr

# 1. Candidate Generation (Llama-3.1-8B-Instruct)

# qrel_file_path="/tmp2/intern/guanwei/DATA/topics/qrecc_train_qrels_binary.json"
# input_path="/tmp2/intern/guanwei/DATA/topics/qrecc_flatten_train_topics.json"
# output_path="/tmp2/intern/guanwei/DATA/candidates/qrecc_train_llama_temp07_cand05_all.jsonl"

# N_candidates=5

# python chat_prompt_RAR_CoT.py \
# --qrel_file_path=$qrel_file_path \
# --test_file_path=$input_path \
# --demo_file_path="./demonstrations.json" \
# --output_path=$output_path \
# --n_generation=$N_candidates \
# --omit_pr