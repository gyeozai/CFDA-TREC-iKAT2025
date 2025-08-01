import argparse
import json
import os
import pickle
import unicodedata
from openai import OpenAI
from tqdm import tqdm

MAX_RETRY = 3
NUM_DIRECT_PASSAGES = 3
SUMMARY_CHUNK_SIZE = 5
RESPONSE_LIMIT = 250

from prompts import (
    SYSTEM_PROMPT_SUMMARIZE,
    USER_PROMPT_SUMMARIZE,
    SYSTEM_PROMPT_RESPONSE,
    USER_PROMPT_RESPONSE
)

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

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

# ========== Function Definitions ==========

def load_lookup_map(path):
    try:
        with open(path, 'rb') as f:
            lookup_map = pickle.load(f)
        print(f"[INFO] Lookup map loaded. Contains {len(lookup_map)} entries.")
        return lookup_map
    except FileNotFoundError:
        print(f"[ERROR] Lookup map file not found at '{path}'")
        return None

def get_passage_text(doc_id, lookup_map):
    location = lookup_map.get(doc_id)
    if not location:
        print(f"[ERROR] ID {doc_id} NOT FOUND in lookup map")
        return None
    
    filepath, offset = location
    try:
        with open(filepath, 'rb') as f:
            f.seek(offset)
            line_bytes = f.readline()
            data = json.loads(line_bytes.decode('utf-8'))
            return data.get('contents') or data.get('text', '')
    except Exception as e:
        print(f"[ERROR] Error reading passage for {doc_id}: {e}")
        return None

def truncate_response(text, limit):
    if not text:
        return ""
    # Normalize and split into words, matching the required length computation
    tokenized = unicodedata.normalize("NFKC", text).split()
    if len(tokenized) > limit:
        # Slice the list of words and join them back into a string
        return " ".join(tokenized[:limit])
    return text

def summarize_passages_with_gpt(passages_to_summarize, context, utterance):
    formatted_passages = "\n\n".join(f"Passage {i+1}:\n{text}" for i, text in enumerate(passages_to_summarize))
    user_content = USER_PROMPT_SUMMARIZE.format(
        context=context or "No history yet.",
        utterance=utterance,
        passages_text=formatted_passages
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] An error occurred during summarization: {e}")
        return "summary:"

def parse_summary_response(response_text):
    if "summary:" in response_text.lower():
        result = response_text.split(":", 1)[1].strip()
        return result if result else "INVALID_FORMAT"
    return "INVALID_FORMAT"

def generate_response_with_gpt(passages, context, ptkb, utterance):
    ptkb_str = "\n".join(f"- {p}" for p in ptkb) if ptkb else "None provided."
    passages_str = "\n\n".join(f"Passage {i+1}:\n{p}" for i, p in enumerate(passages))
    
    user_content = USER_PROMPT_RESPONSE.format(
        passages_text=passages_str,
        context=context or "No history yet.",
        ptkb_list=ptkb_str,
        utterance=utterance
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_RESPONSE},
                {"role": "user", "content": user_content}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] An error occurred during response generation: {e}")
        return "response: Error during API call."

def parse_final_response(response_text):
    if "response:" in response_text.lower():
        result = response_text.split(":", 1)[1].strip()
        return result if result else "INVALID_FORMAT"
    return "INVALID_FORMAT"


# ========== Main Function ==========

def main(args):
    template_data = load_jsonl(args.template_file)
    refer_data = load_json(args.refer_file)
    lookup_map = load_lookup_map(args.lookup_file)
    if not lookup_map: return

    turn_history_map = {}
    for topic in refer_data:
        topic_num = topic['number']
        for turn in topic['responses']:
            full_turn_id = f"{topic_num}_{turn['turn_id']}"
            turn_history_map[full_turn_id] = {
                'user_utterance': turn['user_utterance'],
                'response': turn['response'],
                'resolved_utterance': turn.get('resolved_utterance', turn['user_utterance'])
            }
    
    for item in tqdm(template_data, desc="Generating Responses"):
        turn_id = item['turn_id']
        
        try:
            topic_num, turn_num_str = turn_id.rsplit('_', 1)
            turn_num = int(turn_num_str)
        except ValueError:
            print(f"[ERROR] Invalid turn_id format: {turn_id}")
            continue
        
        # Step 1: Build the conversation context
        context = ""
        for i in range(1, turn_num):
            prev_turn_id = f"{topic_num}_{i}"
            if prev_turn_id in turn_history_map:
                turn_info = turn_history_map[prev_turn_id]
                context += f"USER: {turn_info['user_utterance']}\n"
                
                system_response = turn_info['response'].split()[:25]  # 拿前25個詞
                system_response_truncated = ' '.join(system_response)
                context += f"SYSTEM: {system_response_truncated} ...\n"

        context = context.strip()
        
        current_utterance = turn_history_map[turn_id]['resolved_utterance']

        # Step 2: Extract PTKB and PASSAGE from the item
        for response_obj in item.get('responses', []):
            ptkb_provenance = response_obj.get('ptkb_provenance', [])
            citations = response_obj.get('citations', {})
            
            all_passage_ids = list(citations.keys())
            retrieved_passages_text = []
            for doc_id in all_passage_ids:
                text = get_passage_text(doc_id, lookup_map)
                if text: retrieved_passages_text.append(text)
            
            # Step 3: Format relevant passages
            final_passages = []
            final_passages.extend(retrieved_passages_text[:NUM_DIRECT_PASSAGES])
            
            passages_to_summarize = retrieved_passages_text[NUM_DIRECT_PASSAGES:]
            if passages_to_summarize:
                chunks = []
                i = len(passages_to_summarize)
                while i > 0:
                    start_index = max(0, i - SUMMARY_CHUNK_SIZE)
                    chunks.insert(0, passages_to_summarize[start_index:i])
                    i = start_index

                for chunk in chunks:
                    summary = ""
                    for retry_attempt in range(MAX_RETRY):
                        gpt_summary_response = summarize_passages_with_gpt(
                            chunk, context, current_utterance
                        )
                        summary = parse_summary_response(gpt_summary_response)
                        if summary != "INVALID_FORMAT":
                            break
                        if retry_attempt < MAX_RETRY - 1:
                            print(f"[WARNING] Retrying summarization ({retry_attempt + 2}/{MAX_RETRY})...")
                    
                    final_passages.append(summary)

            # Step 4: Generate the final response
            final_response_text = ""
            for i in range(MAX_RETRY):
                gpt_final_response = generate_response_with_gpt(
                    final_passages, context, ptkb_provenance, current_utterance
                )
                final_response_text = parse_final_response(gpt_final_response)
                if final_response_text != "INVALID_FORMAT":
                    final_response_text = truncate_response(final_response_text, RESPONSE_LIMIT)
                    break
                print(f"[WARNING] Retrying response generation ({i+2}/{MAX_RETRY})...")

            # Step 5: Update the item with the final response
            response_obj['text'] = final_response_text

    output_path = args.output_file if args.output_file else args.template_file
    save_jsonl(template_data, output_path)
    print(f"[INFO] put_response.py: Updated file saved to {output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate personalized responses based on context, PTKB, and retrieved passages.")
    parser.add_argument("--template_file", required=True, help="Path to the input JSONL file with PTKB provenance and citations.")
    parser.add_argument("--refer_file", required=True, help="Path to the reference JSON file with full conversation histories.")
    parser.add_argument("--lookup_file", required=True, help="Path to the passage lookup.pkl file.")
    parser.add_argument("--output_file", help="Path to save the updated file (overwrites template_file if not specified)")
    parser.add_argument("--num_direct_passages", type=int, help="Number of passages to include directly in the response.")
    parser.add_argument("--summary_chunk_size", type=int, help="Number of passages to summarize in each chunk.")
    args = parser.parse_args()
    
    if args.num_direct_passages is not None:
        NUM_DIRECT_PASSAGES = args.num_direct_passages
    if args.summary_chunk_size is not None:
        SUMMARY_CHUNK_SIZE = args.summary_chunk_size
        
    main(args)