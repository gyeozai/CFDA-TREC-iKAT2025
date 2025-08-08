export SPLADE_CONFIG_NAME="config_hf_splade_ikat.yaml"
index_dir=/tmp2/clueweb22/indexes/splade_index

# 5. SPLADE retrieval

eval_queries=[/tmp2/intern/guanwei/DATA/queries/queries_gpt4o_2023.tsv,/tmp2/intern/guanwei/DATA/queries/queries_gpt4o_2024.tsv]
out_dir=EXP/gpt4o_splade

CUDA_VISIBLE_DEVICES=1 python -m splade.retrieve init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
        config.pretrained_no_yamlconfig=true config.index_dir=$index_dir \
        config.out_dir=$out_dir \
        data.Q_COLLECTION_PATH=$eval_queries