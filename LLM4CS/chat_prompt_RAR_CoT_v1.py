import argparse
import json
import os

from chat_promptor import RewriteAndResponsePromptor
from generator_gpt import ChatGenerator
from tqdm import tqdm

# ========== Function Definitions ==========

def get_finished_sample_ids(output_file_path):
    finished_samples = {}
    if os.path.exists(output_file_path):
        with open(output_file_path) as f:
            data = f.readlines()
        for line in data:
            line = json.loads(line)
            finished_samples[line['sample_id']] = {}
            if "predicted_rewrite" in line:
                finished_samples[line['sample_id']]["predicted_rewrite"] = line['predicted_rewrite']
            if "predicted_response" in line:
                finished_samples[line['sample_id']]["predicted_response"] = line['predicted_response']
            if "cot" in line:
                finished_samples[line['sample_id']]["cot"] = line['cot']
            if "rewrite_part_text" in line:
                finished_samples[line['sample_id']]["rewrite_part_text"] = line['rewrite_part_text']

    return finished_samples

def get_has_qrel_label_sample_ids(qrel_file):
    with open(qrel_file, "r") as f:
        qrel_data = json.load(f)

    qids = set(qrel_data.keys())
    return qids

# ========== Main Function ==========

def main():
    parser = argparse.ArgumentParser(description="Run the Candidate Generation of LLM4CS.")
    parser.add_argument("--qrel_file_path", type=str, help="Path to qrel file. If provided, filter samples based on it.")
    parser.add_argument("--test_file_path", type=str, required=True, help="Path to test set file (json).")
    parser.add_argument("--demo_file_path", type=str, required=True, help="Path to demo prompt file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generated outputs.")
    parser.add_argument("--n_generation", type=int, required=True, help="Number of rewrite candidates per query.")
    parser.add_argument("--omit_pr", action="store_true", default=False, help="Omit pseudo-response (used in LLM4CS setting).")
    parser.add_argument("--multiple", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()

    promptor = RewriteAndResponsePromptor(args.demo_file_path, enable_cot=True)
    # promptor = RewriteAndResponsePromptor(args.demo_file_path, enable_cot=False)
    model_kwargs = {
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 256,
        "stop": promptor.stop_tokens
    }
    generator = ChatGenerator(args.n_generation, args.id, **model_kwargs)

    output_path = args.output_path
    out_f = open(output_path, "w")

    finished_samples = get_finished_sample_ids(output_path)
    if args.qrel_file_path:
        has_qrel_labels_samples = get_has_qrel_label_sample_ids(args.qrel_file_path)
   
    with open(args.test_file_path, 'r', encoding='utf-8') as f:
        test_data_dict = json.load(f)

    for sample_id, data in tqdm(test_data_dict.items(), total=len(test_data_dict)):
        
        if sample_id in finished_samples:
            continue
        if args.qrel_file_path and sample_id not in has_qrel_labels_samples:
            continue

        ptkb = data.get("ptkb", "")
        raw_context_str = data.get("context", "")
        current_query = data.get("utterance", "")

        context = []
        if raw_context_str:
            turns = raw_context_str.strip().split('\n')
            for turn in turns:
                if turn.startswith("USER: "):
                    context.append(turn.replace("USER: ", "", 1).strip())
                elif turn.startswith("SYSTEM: "):
                    context.append(turn.replace("SYSTEM: ", "", 1).strip())

        if len(context) % 2 != 0:
            context = context[:-1]

        if len(context) == 0:
            converted_context = None
        else:
            converted_context = []
            for i in range(0, len(context), 2):
                converted_context.append({"question": context[i], "response": context[i+1]})

        current_turn = {"question": current_query}

        prompt = promptor.build_turn_prompt(ptkb, converted_context, current_turn)

        n_outputs = generator.generate(prompt, promptor.parse_returned_text)
        cot_list, rewrite_list, response_list = list(zip(*n_outputs))
        # rewrite_list, response_list = list(zip(*n_outputs, strict=False))

        if args.omit_pr:
            response_list = [""] * len(rewrite_list)
        
        record = {}
        record["sample_id"] = sample_id
        record["context"] = context 
        record["current_query"] = current_query
        record['predicted_rewrite'] = rewrite_list
        record['predicted_response'] = response_list

        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
        
if __name__ == "__main__":
    main()