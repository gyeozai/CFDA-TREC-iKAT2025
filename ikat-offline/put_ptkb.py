import argparse
import json
import os
from openai import OpenAI
from tqdm import tqdm

from prompts import (
    SYSTEM_PROMPT_NEW_PTKB,
    USER_PROMPT_TEMPLATE_NEW_PTKB,
    SYSTEM_PROMPT_RELEVANCE,
    USER_PROMPT_TEMPLATE_RELEVANCE
)

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

MAX_RETRY = 3

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

def get_new_ptkb_from_gpt(context, utterance, ptkb_list):
    ptkb_str = "\n".join([f"- {p}" for p in ptkb_list]) if ptkb_list else "None"
    user_content = USER_PROMPT_TEMPLATE_NEW_PTKB.format(
        context=context or "No history yet.",
        utterance=utterance,
        ptkb_list=ptkb_str
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_NEW_PTKB},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] An error occurred while calling GPT for new PTKB: {e}")
        return "nope"

def parse_new_ptkb_response(response_text):
    if "ptkb:" in response_text.lower():
        result = response_text.split(":", 1)[1].strip()
        if result.lower() == 'nope' or not result:
            return None
        return result
    else:
        print(f"[WARNING] GPT response did not follow the expected format: '{response_text}'")
        return "INVALID_FORMAT"

def get_relevant_ptkbs_from_gpt(context, utterance, ptkb_list):
    if not ptkb_list:
        return "ptkb:\nnope"
    
    ptkb_str = "\n".join([f"- {p}" for p in ptkb_list])
    user_content = USER_PROMPT_TEMPLATE_RELEVANCE.format(
        context=context or "No history yet.",
        utterance=utterance,
        ptkb_list=ptkb_str
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_RELEVANCE},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] An error occurred while calling GPT for relevance classification: {e}")
        return "ptkb:\nnope"

def parse_relevance_response(response_text):
    if "ptkb:" in response_text.lower():
        content = response_text.split(":", 1)[1].strip()
        if not content or content.lower() == 'nope':
            return []
        relevant_ptkbs = [line.strip() for line in content.split('\n') if line.strip()]
        return relevant_ptkbs
    else:
        print(f"[WARNING] Relevance response did not follow format: '{response_text}'")
        return "INVALID_FORMAT"
    
# ========== Main Function ==========

def main(args):
    template_data = load_jsonl(args.template_file)
    refer_data = load_json(args.refer_file)

    topic_to_initial_ptkb = {topic['number']: topic['ptkb'] for topic in refer_data}
    topic_ptkbs = {}

    turn_history_map = {}
    for topic in refer_data:
        topic_num = topic['number']
        for turn in topic['responses']:
            full_turn_id = f"{topic_num}_{turn['turn_id']}"
            turn_history_map[full_turn_id] = {
                'user_utterance': turn['user_utterance'],
                'response': turn['response'],
                'resolved_utterance': turn['resolved_utterance']
            }
            
    for item in tqdm(template_data, desc="Processing Turns"):
        turn_id = item['turn_id']
        
        try:
            topic_num, turn_num_str = turn_id.rsplit('_', 1)
            turn_num = int(turn_num_str)
        except ValueError:
            print(f"[WARNING] Could not parse turn_id '{turn_id}'. Skipping.")
            continue

        if topic_num not in topic_ptkbs:
            topic_ptkbs[topic_num] = list(topic_to_initial_ptkb.get(topic_num, []))

        # Step 1: Build the conversation context
        context = ""
        for i in range(1, turn_num):
            prev_turn_id = f"{topic_num}_{i}"
            if prev_turn_id in turn_history_map:
                turn_info = turn_history_map[prev_turn_id]
                context += f"USER: {turn_info['user_utterance']}\n"
                context += f"SYSTEM: {turn_info['response']}\n"
        context = context.strip()

        # Step 2: Identify NEW PTKB from the current utterance
        current_utterance = turn_history_map[turn_id]['resolved_utterance']
        current_ptkb_list = topic_ptkbs[topic_num]
        
        retry_count = 0
        new_ptkb_statement = None

        while retry_count < MAX_RETRY:
            gpt_response = get_new_ptkb_from_gpt(context, current_utterance, current_ptkb_list)

            result = parse_new_ptkb_response(gpt_response)
            if result == "INVALID_FORMAT":
                retry_count += 1
                continue
            else:
                new_ptkb_statement = result
                break
            
        if new_ptkb_statement:
            topic_ptkbs[topic_num].append(new_ptkb_statement)

        # Step 3: Classify RELEVANT PTKBs
        updated_ptkb_list = topic_ptkbs[topic_num]
        
        retry_count = 0
        relevant_ptkbs = []
        while retry_count < MAX_RETRY:
            gpt_response = get_relevant_ptkbs_from_gpt(context, current_utterance, updated_ptkb_list)
            
            result = parse_relevance_response(gpt_response)
            if result == "INVALID_FORMAT":
                retry_count += 1
                continue
            else:
                relevant_ptkbs = result
                break
        
        # if relevant_ptkbs:
            # print(f"[INFO] Relevant PTKBs for {turn_id}: {relevant_ptkbs}")

        # Step 4: Update the item with new PTKB and relevant PTKBs
        if 'responses' in item and isinstance(item['responses'], list):
            for response_obj in item['responses']:
                response_obj['ptkb_provenance'] = relevant_ptkbs
        else:
            print(f"[WARNING] 'responses' key not found or not a list for {turn_id}. Cannot add ptkb_provenance.")

    output_path = args.output_file if args.output_file else args.template_file
    save_jsonl(template_data, output_path)
    print(f"[INFO] put_ptkb.py: Updated file saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify new PTKB from conversations and update a template file."
    )
    parser.add_argument("--template_file", required=True, help="Path to the input template JSONL file to be updated.")
    parser.add_argument("--refer_file", required=True, help="Path to the reference JSON file with topics and initial PTKBs.")
    parser.add_argument("--output_file", help="Where to save the updated template (overwrite if not set)")
    
    args = parser.parse_args()
    main(args)