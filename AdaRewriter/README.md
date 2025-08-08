# AdaRewriter: Best-of-N Inference

This component is designed for best-of-n inference using the AdaRewriter method, as described in the original AdaRewriter paper. It focuses on improving conversational query reformulation through test-time adaptation.

## Innovations

Our approach differs from the original AdaRewriter method by removing the dense retrieval score term from the scoring function for ranking assessment. This modification aligns with our use of a sparse retrieval system (i.e., SPLADE) for passage retrieval, ensuring consistency and better performance.

## Usage

The script `run_adarewriter.sh` is provided to execute the AdaRewriter process. Follow these steps to use it:

### Training Phase

#### Ranking Assessment

1. **Configure Parameters**:
   - `N_candidates`: Specify the number of candidates to generate. This should align with the `input_path` data.
   - `index_path`: Specify the path to the index used for ranking assessment.
   - `qrel_path`: Specify the path to the qrel file. Ensure the file format matches the provided examples.
   - `input_path`: Specify the path to the input data file.
   - `output_path`: Specify the path to save the ranking results.
2. **Run the Script**: After updating the script and preparing the required data, execute the following command:

    ```bash
    bash run_adarewriter.sh
    ```

#### Training AdaRewriter

1. **Configure Parameters**:
   - `N_candidates`: Specify the number of candidates to generate. This should align with the `input_path` data.
   - `input_path`: Specify the path to the input data file.
   - `--checkpoint` (Optional): Use this flag to specify a checkpoint file to continue training from a specified previous checkpoint.
2. **Run the Script**: After updating the script and preparing the required data, execute the following command:

    ```bash
    bash run_adarewriter.sh
    ```

### Testing Phase

1. **Configure Parameters**:
   - `N_candidates`: Specify the number of candidates to generate. This should align with the `input_path` data.
   - `input_path`: Specify the path to the input data file.
   - `output_path`: Specify the path to save the testing results.
   - `--checkpoint`: Specify the path to the trained model checkpoint to use for testing.
2. **Run the Script**: After updating the script and preparing the required data, execute the following command:

    ```bash
    bash run_adarewriter.sh
    ```

## Citation

If you use this component, please cite the original AdaRewriter paper as follows:

```bibtex
@misc{lai2025adarewriterunleashingpowerpromptingbased,
      title={AdaRewriter: Unleashing the Power of Prompting-based Conversational Query Reformulation via Test-Time Adaptation}, 
      author={Yilong Lai and Jialong Wu and Zhenglin Wang and Deyu Zhou},
      year={2025},
      eprint={2506.01381},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.01381}, 
}
```

## Source Code

The original AdaRewriter source code is available at: [https://anonymous.4open.science/r/AdaRewriter-anonymous-3177](https://anonymous.4open.science/r/AdaRewriter-anonymous-3177)
