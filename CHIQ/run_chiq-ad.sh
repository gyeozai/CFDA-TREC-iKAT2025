# 1. Candidate Generation (gpt-4o-mini)

export OPENAI_API_KEY="your_openai_api_key"

data_file_path="/tmp2/intern/guanwei/DATA/topics/2025_flatten_test_topics.json"
output_path="/tmp2/intern/guanwei/DATA/candidates/ikat2025_test_chiqnpr_temp10_cand10.jsonl"

N_candidates=10

python rewrite_chiq.py \
--data_file_path=$data_file_path \
--output_path=$output_path \
--n_generation=$N_candidates