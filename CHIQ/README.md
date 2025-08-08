# CHIQ-AD: Candidate Generation

This component is designed for candidate generation using the CHIQ-AD method, as described in the original CHIQ paper. It focuses on generating rewritten queries for conversational search tasks.

## Innovations

Our approach differs from the original CHIQ-AD method by removing the History Summarization (HS) step. Based on our experiments, this modification leads to better and more stable performance. The removal of HS avoids the loss of contextual information, which is crucial for generating high-quality rewrites.

## Usage

The script `run_chiq.sh` is provided to execute the candidate generation process. Follow these steps to use it:

1. **Set the OpenAI API Key**: Update the `OPENAI_API_KEY` variable in the script with your OpenAI API key.
2. **Configure Parameters**:
   - `N_candidates`: Specify the number of candidates to generate.
   - `data_file_path`: Specify the path to the input data file. Ensure the file format matches the provided examples.
   - `output_path`: Specify the path to save the generated candidates.

3. **Run the Script**: After updating the script and preparing the required data, execute the following command:

    ```bash
    bash run_chiq.sh
    ```

## Citation

If you use this component, please cite the original CHIQ paper as follows:

```bibtex
@misc{mo2024chiqcontextualhistoryenhancement,
      title={CHIQ: Contextual History Enhancement for Improving Query Rewriting in Conversational Search}, 
      author={Fengran Mo and Abbas Ghaddar and Kelong Mao and Mehdi Rezagholizadeh and Boxing Chen and Qun Liu and Jian-Yun Nie},
      year={2024},
      eprint={2406.05013},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2406.05013}, 
}
```

## Source Code

The original CHIQ source code is available at: [https://github.com/fengranMark/CHIQ](https://github.com/fengranMark/CHIQ)