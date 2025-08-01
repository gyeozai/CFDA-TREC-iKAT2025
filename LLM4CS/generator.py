import time
from openai import OpenAI
from collections import defaultdict

# from https://github.com/texttron/hyde/blob/main/src/hyde/generator.py
class ChatGenerator:
    def __init__(self,
                 n_generation,
                 id=0,
                 **kwargs):
        self.model_name = 'meta-llama/Llama-3.1-8B-Instruct'
        self.api_key = "ollama"
        self.n_generation = n_generation
        self.kwargs = kwargs
        if id == 0:
            self.client = OpenAI(
                api_key="ollama",
                base_url="http://localhost:8000/v1",
            )
        else:
            self.client = OpenAI(
                api_key="ollama",
                base_url="http://localhost:8011/v1",
            )

    def parse_result(self, result, parse_fn):
        choices = result.choices
        n_fail = 0
        res = []

        for i in range(len(choices)):
            output = choices[i].message.content
            output = parse_fn(output)

            if not output:
                n_fail += 1
            else:
                res.append(output)

        return n_fail, res


    def generate(self, prompt, parse_fn):
        """
        Generates responses for a given prompt using streaming.

        This method requests `n` completions in streaming mode, collects them,
        parses them, and retries if some of the generations fail parsing.

        Args:
            prompt (str): The input prompt to send to the model.
            parse_fn (function): A function to parse the raw string output from the model.
                                 It should return the parsed data or None/False if parsing fails.

        Returns:
            list: A list of successfully generated and parsed outputs.
        
        Raises:
            ValueError: If it fails to generate any successful output after 20 retries.
        """
        n_to_generate = self.n_generation
        final_output = []
        n_try = 0

        while len(final_output) < self.n_generation:
            # Safety break after 20 attempts
            if n_try == 20:
                if not final_output:
                    raise ValueError("Have tried 20 times but still only got 0 successful outputs")
                # If some outputs were generated, fill the rest with duplicates to meet the count
                while len(final_output) < self.n_generation:
                    final_output.append(final_output[0])
                break

            completions = defaultdict(str)
            try:
                # Make the API call with stream=True
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"{prompt}"},
                    ],
                    n=n_to_generate,
                    stream=True, # Explicitly enable streaming
                    **self.kwargs
                )

                # Process the stream to assemble the full responses
                for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        # The 'index' identifies which of the 'n' completions this chunk belongs to
                        completions[choice.index] += choice.delta.content or ""

            except Exception as e:
                print(f"An API error occurred: {e}")
                time.sleep(20)
                print("Triggered an error, waiting 20s before retrying...")
                n_try += 1
                continue

            # --- Parsing logic (integrated from the old parse_result method) ---
            n_fail = 0
            successfully_parsed = []
            
            # Iterate through the number of requested completions
            for i in range(n_to_generate):
                # completions.get(i, "") handles cases where a generation might be missing
                full_text = completions.get(i, "")
                if not full_text:
                    n_fail += 1
                    continue

                parsed_item = parse_fn(full_text)
                if not parsed_item:
                    n_fail += 1
                else:
                    successfully_parsed.append(parsed_item)
            
            final_output.extend(successfully_parsed)

            if n_fail == 0:
                # All requested generations were successful
                break
            else:
                # If some failed, update the number to generate for the next retry
                print(f"Failed to parse {n_fail} generations. Retrying for the failed ones.")
                n_to_generate = n_fail
            
            n_try += 1

        return final_output[:self.n_generation]