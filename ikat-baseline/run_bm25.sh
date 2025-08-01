index_path2=/tmp2/clueweb22/indexes/pyserini_index

# 1. BM25 retrival on human rewrite 2023
python -m bm25.retrieve --index_path $index_path2 \
                        --topics /tmp2/intern/guanwei/DATA/queries/queries_manual_2023.tsv \
                        --output EXP/manual_bm25/IKAT2023/run.json

python -m splade.evaluate --run_dir EXP/manual_bm25/IKAT2023/run.json \
                          --qrel_file_path /tmp2/intern/guanwei/DATA/topics/2023_test_qrels.json \
                          --qrel_binary_file_path /tmp2/intern/guanwei/DATA/topics/2023_test_qrels_binary.json

# 2. BM25 retrival on human rewrite 2024

python -m bm25.retrieve --index_path $index_path2 \
                        --topics /tmp2/intern/guanwei/DATA/queries/queries_manual_2024.tsv \
                        --output EXP/manual_bm25/IKAT2024/run.json

python -m splade.evaluate --run_dir EXP/manual_bm25/IKAT2024/run.json \
                          --qrel_file_path /tmp2/intern/guanwei/DATA/topics/2024_test_qrels.json \
                          --qrel_binary_file_path /tmp2/intern/guanwei/DATA/topics/2024_test_qrels_binary.json

#3. BM25 retrieval on GPT4o rewrite 2023

python -m bm25.retrieve --index_path $index_path2 \
                        --topics /tmp2/intern/guanwei/DATA/queries/queries_gpt4o_2023.tsv \
                        --output EXP/gpt4o_bm25/IKAT2023/run.json

python -m splade.evaluate --run_dir EXP/gpt4o_bm25/IKAT2023/run.json \
                          --qrel_file_path /tmp2/intern/guanwei/DATA/topics/2023_test_qrels.json \
                          --qrel_binary_file_path /tmp2/intern/guanwei/DATA/topics/2023_test_qrels_binary.json

# 4. BM25 retrieval on GPT4o rewrite 2024

python -m bm25.retrieve --index_path $index_path2 \
                        --topics /tmp2/intern/guanwei/DATA/queries/queries_gpt4o_2024.tsv \
                        --output EXP/gpt4o_bm25/IKAT2024/run.json

python -m splade.evaluate --run_dir EXP/gpt4o_bm25/IKAT2024/run.json \
                          --qrel_file_path /tmp2/intern/guanwei/DATA/topics/2024_test_qrels.json \
                          --qrel_binary_file_path /tmp2/intern/guanwei/DATA/topics/2024_test_qrels_binary.json
