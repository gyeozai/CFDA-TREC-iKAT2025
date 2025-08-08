# iKAT-Offline: PTKB Classification & Response Generation

This component is designed for PTKB statement classification and response generation as part of the iKAT-Offline pipeline. It automates the process of generating responses for conversational search tasks.

## Usage

The script `run_offline_auto.sh` is provided to generate the submission of  the automatic task. Follow these steps to use it:

### Automatic Task

1. **Configure Parameters**:
   - `lookup_file`: Specify the path to the lookup file for passage retrieval.
   - `retrieve_file`: Specify the path to the retrieved passage file.
   - `refer_file`: Specify the path to the reference topics file.
   - `template_file`: Specify the path to save the generated template file.
   - `team_id`: Set the team identifier.
   - `run_id`: Set the run identifier.

2. **Pipeline Overview**:
   - The pipeline extracts up to `num_passages` citations from the references, including only passages with a score greater than `score_threshold`.
   - For response generation, citations are used as knowledge or facts to answer the utterance. Since all direct passages may be too lengthy to form the prompt, the process includes compression:
     - `num_direct_passages` direct passages are retained.
     - The remaining passages are summarized, with every `summary_chunk_size` passages condensed into a single passage to improve response quality.

3. **Run the Script**: After updating the script, execute the following command:

    ```bash
    bash run_offline_auto.sh
    ```

### Generation Only Task

The process is similar to the automatic task. To execute, run the following command:

```bash
bash run_offline_gen-only.sh
```