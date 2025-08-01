export SPLADE_CONFIG_NAME="config_hf_splade_ikat.yaml"
index_dir=/tmp2/clueweb22/indexes/splade_index
index_path2=/tmp2/clueweb22/indexes/pyserini_index

# 1. SPLADE retrieval on human rewrite

eval_queries=[/tmp2/intern/guanwei/DATA/queries/queries_manual_2023.tsv,/tmp2/intern/guanwei/DATA/queries/queries_manual_2024.tsv]
out_dir=EXP/manual_splade

CUDA_VISIBLE_DEVICES=1 python -m splade.retrieve init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
        config.pretrained_no_yamlconfig=true config.index_dir=$index_dir \
        config.out_dir=$out_dir \
        data.Q_COLLECTION_PATH=$eval_queries 

# 2. SPLADE retrieval on GPT4o-mini rewrite

eval_queries=[/tmp2/intern/guanwei/DATA/queries/queries_gpt4o_2023.tsv,/tmp2/intern/guanwei/DATA/queries/queries_gpt4o_2024.tsv]
out_dir=EXP/gpt4o_splade

CUDA_VISIBLE_DEVICES=1 python -m splade.retrieve init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
        config.pretrained_no_yamlconfig=true config.index_dir=$index_dir \
        config.out_dir=$out_dir \
        data.Q_COLLECTION_PATH=$eval_queries

# 3. Rerank with DeBERTa_v3

CUDA_VISIBLE_DEVICES=1 python -m rerank.rerank \
                        --index_path $index_path2 \
                        --model naver/trecdl22-crossencoder-debertav3 \
                        --query_file /tmp2/intern/guanwei/DATA/queries/queries_gpt4o_2024.tsv \
                        --run EXP/gpt4o_splade/IKAT2024/run.json \
                        --output EXP/gpt4o_splade_rerank_deberta/IKAT2024/run.json

CUDA_VISIBLE_DEVICES=1 python -m splade.evaluate \
                        --run_dir EXP/gpt4o_splade_rerank_deberta/IKAT2024/run.json \
                        --qrel_file_path /tmp2/intern/guanwei/DATA/topics/2024_test_qrels.json \
                        --qrel_binary_file_path /tmp2/intern/guanwei/DATA/topics/2024_test_qrels_binary.json

# CUDA_VISIBLE_DEVICES=1 python -m rerank.rerank \
#                         --index_path $index_path2 \
#                         --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
#                         --query_file /tmp2/intern/guanwei/DATA/queries/queries_gpt4o_2024.tsv \
#                         --run EXP/gpt4o_splade/IKAT2024/run.json \
#                         --output EXP/gpt4o_splade_rerank_minilm/IKAT2024/run.json

# CUDA_VISIBLE_DEVICES=1 python -m splade.evaluate \
#                         --run_dir EXP/gpt4o_splade_rerank_minilm/IKAT2024/run.json \
#                         --qrel_file_path /tmp2/intern/guanwei/DATA/topics/2024_test_qrels.json \
#                         --qrel_binary_file_path /tmp2/intern/guanwei/DATA/topics/2024_test_qrels_binary.json
