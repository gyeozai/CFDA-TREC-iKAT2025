export OPENAI_API_KEY="your_openai_api_key"

lookup_file="/tmp2/QReCC/passage_lookup.pkl"
retrieve_file="/tmp2/intern/guanwei/ikat-baseline/EXP/chiq_llmcs_fused_rrf_rerank/IKAT2025/adaR13_chiqnpr_llm4csnpr_run.json"
refer_file="/tmp2/intern/guanwei/DATA/topics/2025_test_topics.json"
template_file="submissions/cfda-ikat2025-auto-5.jsonl"

team_id="cfda"
run_id="auto_npr_npsg20_thru03_d3c5"
num_passages=20
score_threshold=0.3
num_direct_passages=3
summary_chunk_size=5

# 1. Generate the template jsonl file for offline submission

python gen_template.py \
  --team_id ${team_id} \
  --run_id ${run_id} \
  --run_type automatic \
  --refer_file ${refer_file} \
  --output ${template_file}

# 2. Fill the template with the references

python put_reference.py \
  --template_file ${template_file} \
  --retrieved_file ${retrieve_file} \
  --num_passages ${num_passages} \
  --score_threshold ${score_threshold}
  # --output_file submissions/tmp_with_refs.jsonl

# 3. Fill the template with the ptkb_provenance

python put_ptkb.py \
    --template_file ${template_file} \
    --refer_file ${refer_file}
    # --output_file submissions/tmp_with_ptkb.jsonl

# 4. Fill the template with the responses.text

python put_response.py \
  --template_file ${template_file} \
  --refer_file ${refer_file} \
  --lookup_file ${lookup_file} \
  --num_direct_passages ${num_direct_passages} \
  --summary_chunk_size ${summary_chunk_size}
  # --output_file submissions/tmp_with_response.jsonl

# 5. Validate the generated file

python DATA/validate_trec_ikat25.py \
  --input ${template_file} \
  --topics ${refer_file}