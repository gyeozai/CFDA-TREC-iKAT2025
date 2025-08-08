export OPENAI_API_KEY="your_openai_api_key"

lookup_file="/tmp2/QReCC/passage_lookup.pkl"
retrieve_file="/tmp2/intern/guanwei/ikat-offline/DATA/ikat25_gen-only_run.json"
refer_file="/tmp2/intern/guanwei/DATA/topics/2025_test_topics.json"
template_file="submissions/cfda-ikat2025-gen-only-3.jsonl"

team_id="cfda"
run_id="gen-only_npsg20_thru03_d3c5"
num_passages=20
score_threshold=0.3
num_direct_passages=3
summary_chunk_size=5

# 1. Generate the template jsonl file for offline submission

python gen_template.py \
  --team_id ${team_id} \
  --run_id ${run_id} \
  --run_type generation-only \
  --refer_file ${refer_file} \
  --output ${template_file}

# 2. Fill the template with the references

python put_reference.py \
  --template_file ${template_file} \
  --retrieved_file ${retrieve_file} \
  --num_passages ${num_passages} \
  --score_threshold ${score_threshold}
  # --output_file submissions/tmp_with_refs.jsonl

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