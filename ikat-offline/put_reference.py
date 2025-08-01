import argparse
import json
from tqdm import tqdm

SCORE_THRESHOLD = 0
NUM_PASSAGES = 20

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# ========== Main Function ==========

def main(args):
    retrieved_data = load_json(args.retrieved_file)
    template_data = load_jsonl(args.template_file)

    for item in tqdm(template_data, desc="Injecting references and citations"):
        tid = item.get("turn_id")
        
        sorted_references = {}
        selected_citations = {}

        if tid in retrieved_data:
            raw_refs = retrieved_data[tid]

            # Step 1: Sort the retrieved passages by score in descending order
            sorted_refs_list = sorted(raw_refs.items(), key=lambda x: x[1], reverse=True)
            
            sorted_references = dict(sorted_refs_list)

            # Step 2: Select passages for citations based on score
            # Rule: The top-1 ranked passage is always selected.
            if sorted_refs_list:
                top_1_doc, top_1_score = sorted_refs_list[0]
                selected_citations[top_1_doc] = top_1_score
            
            # Rule: Select from top-2 to top-<NUM_PASSAGES> ranked passages
            for doc_id, score in sorted_refs_list[1:NUM_PASSAGES]:
                if score > SCORE_THRESHOLD:
                    selected_citations[doc_id] = score
                else:
                    break 

        else:
            print(f"[WARNING] turn_id {tid} not in retrieved_file. Setting empty references and citations.")

        # Step 3: Inject the sorted references and citations into the item
        # if args.run_type == "automatic":
        item["references"] = sorted_references
        
        if "responses" in item and isinstance(item["responses"], list):
            for response in item["responses"]:
                response["citations"] = selected_citations

    output_path = args.output_file if args.output_file else args.template_file
    save_jsonl(template_data, output_path)
    print(f"[INFO] put_reference.py: Updated file saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject sorted references and citations into a template file.")
    # parser.add_argument("--run_type", required=True, choices=['automatic', 'generation-only'], help="Type of the run.")
    parser.add_argument("--template_file", required=True, help="Path to template JSONL file")
    parser.add_argument("--retrieved_file", required=True, help="Path to run.json (retrieved passages + scores)")
    parser.add_argument("--output_file", help="Path to save the updated file (overwrites template_file if not specified)")
    parser.add_argument("--score_threshold", type=float, help="Score threshold for selecting passages")
    parser.add_argument("--num_passages", type=int, help="Number of passages to consider for citations")
    args = parser.parse_args()
    
    if args.score_threshold is not None:
        SCORE_THRESHOLD = args.score_threshold
    if args.num_passages is not None:
        NUM_PASSAGES = args.num_passages
    
    main(args)