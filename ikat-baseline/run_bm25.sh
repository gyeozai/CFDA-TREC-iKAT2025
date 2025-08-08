index_path2=/tmp2/clueweb22/indexes/pyserini_index

# 5. BM25 retrieval
query_path="/tmp2/intern/guanwei/DATA/queries/queries_gpt4o_2024.tsv"
output_path="/tmp2/intern/guanwei/ikat-baseline/EXP/gpt4o_bm25/IKAT2024/run.json"
qrel_path="/tmp2/intern/guanwei/DATA/topics/2024_test_qrels.json"
qrel_binary_path="/tmp2/intern/guanwei/DATA/topics/2024_test_qrels_binary.json"

python -m bm25.retrieve --index_path $index_path2 \
                        --topics $query_path \
                        --output $output_path

python -m splade.evaluate --run_dir $output_path \
                          --qrel_file_path $qrel_path \
                          --qrel_binary_file_path $qrel_binary_path
