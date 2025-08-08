# LLM4CS: Candidate Generation

This component is designed for candidate generation using the LLM4CS method, specifically the RAR prompting method with Chain-of-Thought (CoT) reasoning, as described in the original LLM4CS paper. It focuses on generating rewritten queries for conversational search tasks.

## Innovations

Our approach differs from the original LLM4CS method by removing the Pseudo Response (PR) step. In the original method, the rewrite is concatenated with the response to form the final query. Based on our experiments, omitting this step results in better and more stable performance, although the exact reason remains unclear.

## Usage

The script `run_llm4cs.sh` is provided to execute the candidate generation process. Follow these steps to use it:

### For GPT-4o-mini

1. **Set the OpenAI API Key**: Update the `OPENAI_API_KEY` variable in the script with your OpenAI API key.
2. **Configure Parameters**:
   - `N_candidates`: Specify the number of candidates to generate.
   - `input_path`: Specify the path to the input data file. Ensure the file format matches the provided examples.
   - `output_path`: Specify the path to save the generated candidates.
3. **Run the Script**: After updating the script and preparing the required data, execute the following command:

    ```bash
    bash run_llm4cs.sh
    ```

### For Llama-3.1-8B-Instruct

1. **Set Up the Model**: Deploy the model from Hugging Face and start the server using the following command:

    ```bash
    vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto --max-model-len 6000 --port 8000 --tensor_parallel_size 1
    ```

2. **Configure Parameters**:
   - `N_candidates`: Specify the number of candidates to generate.
   - `input_path`: Specify the path to the input data file. Ensure the file format matches the provided examples.
   - `output_path`: Specify the path to save the generated candidates.
   - `--qrel_file_path` (Recommended): Use this flag to skip turns without truth passages in the dataset. This is useful for datasets like scai-qrecc-21, where some turns lack truth passages and do not require candidate generation.

3. **Run the Script**: After setting up the server and preparing the required data, execute the following command:

    ```bash
    bash run_llm4cs.sh
    ```

## Citation

If you use this component, please cite the original LLM4CS paper as follows:

```bibtex
@misc{mao2023largelanguagemodelsknow,
      title={Large Language Models Know Your Contextual Search Intent: A Prompting Framework for Conversational Search}, 
      author={Kelong Mao and Zhicheng Dou and Fengran Mo and Jiewen Hou and Haonan Chen and Hongjin Qian},
      year={2023},
      eprint={2303.06573},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2303.06573}, 
}
```

## Source Code

The original LLM4CS source code is available at: [https://github.com/kyriemao/LLM4CS](https://github.com/kyriemao/LLM4CS)
