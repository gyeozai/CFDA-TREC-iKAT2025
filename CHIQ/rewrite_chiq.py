from openai import OpenAI
import json
from tqdm import tqdm
import os
import argparse

from prompts import (
    FILTER_PREFERENCES_PROMPT,
    QD_PROMPT,
    RE_PROMPT,
    PR_PROMPT,
    TS_PROMPT,
    HS_PROMPT,
    REWRITE_PROMPT,
    RESPONSE_PROMPT
)

API_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_key)

# =============== Function Definitions ===============

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

# =============== CHIQ Rewrite Function ===============

def chiq_rewrite(data, n_generation):
    ptkb = data.get("ptkb", "").replace(",", "\n")
    ctx = data.get("context", "")
    user_utterance = data.get("utterance", "")

    topic_switch = TS(ptkb, ctx, user_utterance)

    if topic_switch == "new_topic":
        before, sep, after = ctx.rpartition("USER: ")
        history = "USER: " + after if after else "N/A"
    else:
        history = ctx if ctx else "N/A"

    disambiguated_question = QD(ptkb, history, user_utterance)

    if topic_switch == "new_topic":
        refined_answer = ""
        pseudo_response = ""
    else:
        refined_answer = RE(ptkb, history)
        pseudo_response = PR(ptkb, history, user_utterance)

    if "\nSYSTEM" in history and refined_answer.strip():
        before, sep, after = history.rpartition("\nSYSTEM")
        history = before + "\nSYSTEM: " + refined_answer

    enhanced_history = history
    preference = FilterPreferences(ptkb, enhanced_history, disambiguated_question)

    rewritten_query = Rewrite(
        ptkb,
        enhanced_history,
        disambiguated_question,
        pseudo_response,
        preference,
        n_generation
    )

    return rewritten_query

def FilterPreferences(ptkb, history, question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a reasoning agent that extracts relevant preferences from user profiles for search personalization."},
            {"role": "user", "content": FILTER_PREFERENCES_PROMPT.format(ptkb=ptkb, history=history, question=question)}
        ]
    )
    return response.choices[0].message.content.strip()

def QD(ptkb, history, User_Question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for conversational query rewriting. Follow the instructions carefully and respond in the required format."},
            {"role": "user", "content": QD_PROMPT.format(ptkb=ptkb, history=history, User_Question=User_Question)}
        ]
    )
    return response.choices[0].message.content.strip()

def RE(ptkb, history):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for expanding system responses in a conversation to make them more informative and specific."},
            {"role": "user", "content": RE_PROMPT.format(ptkb=ptkb, history=history)}
        ]
    )
    return response.choices[0].message.content.strip()

def PR(ptkb, history, User_Question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides short, plausible pseudo-responses for conversational search tasks. Keep responses realistic but do not hallucinate new facts."},
            {"role": "user", "content": PR_PROMPT.format(ptkb=ptkb, history=history, User_Question=User_Question)}
        ]
    )
    return response.choices[0].message.content.strip()

def TS(ptkb, history, User_Question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a classification assistant that determines whether a new user query is topically related to previous conversation history."},
            {"role": "user", "content": TS_PROMPT.format(ptkb=ptkb, history=history, User_Question=User_Question)}
        ]
    )
    return response.choices[0].message.content.strip()

def HS(ptkb, enhanced_history):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a summarization assistant specialized in compressing conversational history into concise and accurate summaries."},
            {"role": "user", "content": HS_PROMPT.format(ptkb=ptkb, enhanced_history=enhanced_history)}
        ]
    )
    return response.choices[0].message.content.strip()

def Rewrite(ptkb, enhanced_history, disambiguated_question, pseudo_response, preference, n_generation):
    model_kwargs = {
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 256
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert assistant for query rewriting in conversational search. "
                    "Your task is to rewrite a user’s clarified question into a concise, fully self-contained search query, "
                    "based on the conversation history and user background information.\n\n"
                    "Make sure to:\n"
                    "- Include relevant user preferences from the background if they help focus or personalize the search intent "
                    "(e.g., interests in art, cooking, history, hiking, etc).\n"
                    "- Use the conversation history to disambiguate vague terms like 'there', 'this', 'them', etc.\n"
                    "- Keep the query natural and focused. Do NOT invent new facts.\n"
                    "- Ignore the pseudo response content (it is only a reference for output format).\n"
                    "Output a JSON object like: {\"query\": \"...\"} with no explanation."
                )
            },
            {
                "role": "user",
                "content": REWRITE_PROMPT.format(
                    ptkb=ptkb,
                    enhanced_history=enhanced_history,
                    disambiguated_question=disambiguated_question,
                    pseudo_response=pseudo_response,
                    preference=preference
                )
            }
        ],
        n=n_generation,
        **model_kwargs
    )

    rewritten_queries = []
    for choice in response.choices:
        response_content = choice.message.content.strip()
        try:
            response_json = json.loads(response_content)
            rewritten_queries.append(response_json["query"])
        except (json.JSONDecodeError, KeyError):
            rewritten_queries.append(disambiguated_question)

    return rewritten_queries

def Response(context, ptkb, query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful search assistant that provides direct, factual answers to user queries."},
            {"role": "user", "content": RESPONSE_PROMPT.format(ptkb=ptkb, context=context, query=query)}
        ]
    )
    return response.choices[0].message.content.strip()

# =============== Main Function ===============

def main():
    parser = argparse.ArgumentParser(description="Run the Candidate Generation of CHIQ-AD.")
    parser.add_argument("--qrel_file_path", type=str, help="Path to qrel file. If provided, filter samples based on it.")
    parser.add_argument("--data_file_path", type=str, required=True, help="Path to test set file (json).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generated outputs.")
    parser.add_argument("--n_generation", type=int, required=True, help='the number for generation')
    args = parser.parse_args()

    with open(args.data_file_path, 'r', encoding='utf-8') as f:
        test_data_dict = json.load(f)

    out_f = open(args.output_path, "w")
    finished_samples = get_finished_sample_ids(args.output_path)
    if args.qrel_file_path:
        has_qrel_labels_samples = get_has_qrel_label_sample_ids(args.qrel_file_path)
    
    for sample_id, data in tqdm(test_data_dict.items(), total=len(test_data_dict)):
        
        if sample_id in finished_samples:
            continue
        if args.qrel_file_path and sample_id not in has_qrel_labels_samples:
            continue

        ptkb = data.get("ptkb", "")
        context = data.get("context", "")
        current_query = data.get("utterance", "")

        # 1. Generate the rewrite
        predicted_rewrites = chiq_rewrite(data, args.n_generation)

        context_list = []
        if context:
            turns = context.strip().split('\n')
            for turn in turns:
                if turn.startswith("USER: "):
                    context_list.append(turn.replace("USER: ", "", 1).strip())
                elif turn.startswith("SYSTEM: "):
                    context_list.append(turn.replace("SYSTEM: ", "", 1).strip())

        # 2. Generate the response for each rewrite
        predicted_responses = [""] * len(predicted_rewrites)
                
        # predicted_responses = []
        # for rewrite in predicted_rewrites:
        #     response = Response(context, ptkb, rewrite)
        #     predicted_responses.append(response)

        record = {
            "sample_id": sample_id,
            "context": context_list,
            "current_query": current_query,
            "predicted_rewrite": predicted_rewrites,
            "predicted_response": predicted_responses
        }

        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

    print(f"\nCandidate rewrites saved to {args.output_path}")

if __name__ == "__main__":
    main()
