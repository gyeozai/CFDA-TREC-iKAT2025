import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate template JSONL file.")
    parser.add_argument("--team_id", required=True, help="Your team identifier.")
    parser.add_argument("--run_id", required=True, help="A unique identifier for this run.")
    parser.add_argument("--run_type", required=True, choices=['automatic', 'generation-only'], help="Type of the run.")
    parser.add_argument("--refer_file", required=True, help="The input JSON file (e.g., test_topic.json).")
    parser.add_argument("--output", default="template.jsonl", help="The name of the output JSONL file.")
    args = parser.parse_args()

    with open(args.refer_file, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    with open(args.output, "w", encoding="utf-8") as fout:
        for dialogue in data:
            number = dialogue["number"]
            responses = dialogue.get("responses", [])

            for resp in responses:
                turn_index = resp["turn_id"]
                turn_id = f"{number}_{turn_index}"

                ptkb_provenance_data = []
                if args.run_type == "generation-only":
                    ptkb_provenance_data = resp.get("relevant_ptkbs", [])

                output_obj = {
                    "metadata": {
                        "team_id": args.team_id,
                        "run_id": args.run_id,
                        "run_type": args.run_type
                    },
                    "turn_id": turn_id,
                    "responses": [
                        {
                            "rank": 1,
                            "text": "",
                            "citations": {},
                            "ptkb_provenance": ptkb_provenance_data
                        }
                    ],
                    "references": {}
                }

                fout.write(json.dumps(output_obj) + "\n")

if __name__ == "__main__":
    main()