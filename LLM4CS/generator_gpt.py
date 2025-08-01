import time
import os
from openai import OpenAI

API_key = os.getenv("OPENAI_API_KEY")

# from https://github.com/texttron/hyde/blob/main/src/hyde/generator.py
class ChatGenerator:
    def __init__(self, n_generation, id=0, **kwargs):
        self.model_name = 'gpt-4o-mini'
        self.n_generation = n_generation
        self.kwargs = kwargs
        self.client = OpenAI(api_key=API_key)

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
        n_generation = self.n_generation
        output = []
        n_try = 0

        while True:
            if n_try == 20:
                if len(output) == 0:
                    raise ValueError("Have tried 20 times but still only got 0 successful outputs")
                output += output[:5 - len(output)]
                break

            while True:
                try:
                    result = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"{prompt}"}
                        ],
                        n=n_generation,
                        **self.kwargs
                    )
                    break
                except Exception as e:
                    print(e)
                    time.sleep(20)
                    print("Trigger RateLimitError, wait 20s...")

            n_fail, res = self.parse_result(result, parse_fn)
            output += res

            if n_fail == 0:
                return output
            else:
                n_generation = n_fail

            n_try += 1
