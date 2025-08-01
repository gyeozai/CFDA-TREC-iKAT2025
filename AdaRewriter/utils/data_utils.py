import argparse
import json
import random
from typing import Literal, TypeAlias, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

random.seed(42)

InputTypes: TypeAlias = Literal["direct", "contextual"]


def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids

    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length

    return input_ids, attention_mask

class AdapterDataset(Dataset):
    def __init__(self, args, tokenizer, file_name: str="", use_data_percent: float=1.0, input_type: InputTypes="direct", is_test=False):
        super().__init__()
        self.examples = []
        self.args = args

        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        self.is_test = is_test

        with open(file_name, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)
        n = int(n * use_data_percent)
        if n < len(data):
            data = random.sample(data, n)

        for line in tqdm(data):
            line = line.strip()
            record = json.loads(line)
            if input_type == "direct":
                candidates_ids = []
                sample_id = record["sample_id"]
                if self.is_test is False:
                    ranks = list(record["ranks"])

                    for rewrite, response in zip(record["predicted_rewrite"], record["predicted_response"], strict=False):
                        ids = self.bert_encode(rewrite, response)
                        candidates_ids.append(ids)
                    self.examples.append([sample_id, self.bert_pad(candidates_ids), ranks])
                else:
                    reformulation = []
                    for rewrite, response in zip(record["predicted_rewrite"], record["predicted_response"], strict=False):
                        ids = self.bert_encode(rewrite, response)
                        candidates_ids.append(ids)
                        reformulation.append(f"{rewrite} {response}")
                    self.examples.append([sample_id, self.bert_pad(candidates_ids), reformulation])

            elif input_type == "contextual":
                raise NotImplementedError

    def bert_pad(self, X):
        max_len = self.args.max_len
        if max_len < 0:
            max_len = max(len(x) for x in X)
        result = []
        for x in X:
            if len(x) < max_len:
                x.extend([self.pad_token_id] * (max_len - len(x)))
            result.append(x)
        return result

    def bert_encode(self, rewrite: str, response: str):
        rewrite_ids = self.tokenizer.encode(rewrite, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        ids = [self.cls_token_id]

        ids.extend(rewrite_ids[:self.args.max_len -len(ids) -3])
        ids.append(self.sep_token_id)
        ids.extend(response_ids[:self.args.max_len - len(ids) -3])
        ids.append(self.sep_token_id)

        return ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        def collate_fn(batch: list):
            if args.test is True:
                collated_dict = {
                    "sample_ids": [],
                    "candidates_ids": [],
                    "reformulation": [],
                }
                for sample_id, candidates_ids, reformulation in batch:
                    collated_dict["sample_ids"].append(sample_id)
                    collated_dict["candidates_ids"].append(candidates_ids)
                    collated_dict["reformulation"].append(reformulation)
            else:
                collated_dict = {
                    "sample_ids": [],
                    "candidates_ids": [],
                    "ranks": [],
                }
                for sample_id, candidates_ids, ranks in batch:
                    collated_dict["sample_ids"].append(sample_id)
                    collated_dict["candidates_ids"].append(candidates_ids)
                    collated_dict["ranks"].append(ranks)

            not_need_to_tensor_keys = ["sample_ids", "reformulation"]
            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long) # type: ignore

            return collated_dict

        return collate_fn

class AdapterDatasetWithContext(Dataset):
    def __init__(self, args, tokenizer, file_name: str="", use_data_percent: float=1.0, input_type: InputTypes="direct", is_test: bool=False, max_query_len: int=128, max_concat_length=512):
        super().__init__()
        self.examples = []
        self.args = args
        self.max_query_len = max_query_len
        self.max_concat_length = max_concat_length

        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id


        self.is_test = is_test

        with open(file_name, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)
        n = int(n * use_data_percent)
        if n < len(data):
            data = random.sample(data, n)

        for line in tqdm(data):
            line = line.strip()
            record = json.loads(line)
            if input_type == "direct":
                candidates_ids = []
                sample_id = record["sample_id"]

                if self.is_test is False:
                    ranks = list(record["ranks"])
                    context = record["context"]

                    for rewrite, response in zip(record["predicted_rewrite"], record["predicted_response"], strict=False):
                        # ids = self.bert_encode(rewrite, response)
                        reformulation = f"question: {rewrite} {response}"
                        # reformulation = f"question: {rewrite}"
                        first_context = True

                        input_ids = []
                        tokenizer_query = self.tokenizer.encode(reformulation, add_special_tokens=True, max_length=self.max_query_len)
                        input_ids.extend(tokenizer_query)

                        for j in range(len(context)-1, -1, -1):
                            if j % 2 == 1:
                                max_length = 64 ## response_length
                            else:
                                max_length = 64 ## query_length

                            cur_utt = context[j]
                            if first_context:
                                cur_utt = "context: " + cur_utt
                                first_context = False

                            utt = self.tokenizer.encode(cur_utt, add_special_tokens=False, max_length=max_length, truncation=True)
                            if (len(input_ids) + len(utt) + 1) > self.max_concat_length:
                                input_ids += utt[:(self.max_concat_length - len(input_ids) - 1)] + [self.sep_token_id] # must ended with [SEP]
                                break
                            else:
                                utt = utt + [self.sep_token_id]
                                input_ids.extend(utt)

                        input_ids, _ = padding_seq_to_same_length(input_ids, self.max_concat_length, self.pad_token_id)
                        candidates_ids.append(input_ids)

                    # self.examples.append([sample_id, self.bert_pad(candidates_ids), ranks])
                    self.examples.append([sample_id, candidates_ids, ranks])
                else:
                    reformulation_text = []
                    context = record["context"]

                    for rewrite, response in zip(record["predicted_rewrite"], record["predicted_response"], strict=False):
                        # ids = self.bert_encode(rewrite, response)
                        reformulation = f"question: {rewrite} {response}"
                        # reformulation = f"question: {rewrite}"
                        first_context = True

                        input_ids = []
                        tokenizer_query = self.tokenizer.encode(reformulation, add_special_tokens=True, max_length=self.max_query_len)
                        input_ids.extend(tokenizer_query)

                        for j in range(len(context)-1, -1, -1):
                            if j % 2 == 1:
                                max_length = 64 ## response_length
                            else:
                                max_length = 64 ## query_length

                            cur_utt = context[j]
                            if first_context:
                                cur_utt = "context: " + cur_utt
                                first_context = False

                            utt = self.tokenizer.encode(cur_utt, add_special_tokens=False, max_length=max_length, truncation=True)
                            if (len(input_ids) + len(utt) + 1) > self.max_concat_length:
                                input_ids += utt[:self.max_concat_length - len(input_ids) - 1] + [self.sep_token_id] # must ended with [SEP]
                                break
                            else:
                                utt = utt + [self.sep_token_id]
                                input_ids.extend(utt)

                        input_ids, _ = padding_seq_to_same_length(input_ids, self.max_concat_length)

                        candidates_ids.append(input_ids)
                        reformulation_text.append(f"{rewrite} {response}")

                    # self.examples.append([sample_id, self.bert_pad(candidates_ids), reformulation_text])
                    self.examples.append([sample_id, candidates_ids, reformulation_text])

            elif input_type == "contextual":
                raise NotImplementedError

    def bert_pad(self, X):
        max_len = self.max_concat_length
        if max_len < 0:
            max_len = max(len(x) for x in X)
        result = []
        for x in X:
            if len(x) < max_len:
                x.extend([self.pad_token_id] * (max_len - len(x)))
            result.append(x)
        return result

    def bert_encode(self, rewrite: str, response: str):
        rewrite_ids = self.tokenizer.encode(rewrite, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        ids = [self.cls_token_id]

        ids.extend(rewrite_ids[:self.args.max_len -len(ids) -3])
        ids.append(self.sep_token_id)
        ids.extend(response_ids[:self.args.max_len - len(ids) -3])
        ids.append(self.sep_token_id)

        return ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        def collate_fn(batch: list):
            if args.test is True:
                collated_dict = {
                    "sample_ids": [],
                    "candidates_ids": [],
                    "reformulation": [],
                }
                for sample_id, candidates_ids, reformulation in batch:
                    collated_dict["sample_ids"].append(sample_id)
                    collated_dict["candidates_ids"].append(candidates_ids)
                    collated_dict["reformulation"].append(reformulation)
            else:
                collated_dict = {
                    "sample_ids": [],
                    "candidates_ids": [],
                    "ranks": [],
                }
                for sample_id, candidates_ids, ranks in batch:
                    collated_dict["sample_ids"].append(sample_id)
                    collated_dict["candidates_ids"].append(candidates_ids)
                    collated_dict["ranks"].append(ranks)

            not_need_to_tensor_keys = ["sample_ids", "reformulation"]
            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long) # type: ignore

            return collated_dict

        return collate_fn

class AdapterDatasetWithContextV2(Dataset):
    def __init__(self, args, tokenizer, file_name: str="", use_data_percent: float=1.0, input_type: InputTypes="direct", is_test: bool=False, max_query_len: int=128, max_concat_length=512):
        super().__init__()
        self.examples = []
        self.args = args
        self.max_query_len = max_query_len
        self.max_concat_length = max_concat_length

        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id


        self.is_test = is_test

        with open(file_name, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)
        n = int(n * use_data_percent)
        if n < len(data):
            data = random.sample(data, n)

        for line in tqdm(data):
            line = line.strip()
            record = json.loads(line)
            if input_type == "direct":
                candidates_ids = []
                sample_id = record["sample_id"]

                if self.is_test is False:
                    ranks = list(record["ranks"])
                    context = record["context"]
                    query = record["current_query"]

                    for rewrite, response in zip(record["predicted_rewrite"], record["predicted_response"], strict=False):
                        # ids = self.bert_encode(rewrite, response)
                        reformulation = f"rewrite: {rewrite} {response}"
                        # reformulation = f"rewrite: {rewrite}"
                        # reformulation = f"question: {rewrite}"
                        query = f"query: {query}"
                        first_context = True

                        input_ids = []
                        tokenized_rewrite = self.tokenizer.encode(reformulation, add_special_tokens=True, max_length=self.max_query_len)

                        input_ids.extend(tokenized_rewrite)
                        # input_ids.append(self.sep_token_id)

                        tokenized_query = self.tokenizer.encode(query, add_special_tokens=False, max_length=self.max_query_len)
                        input_ids.extend(tokenized_query)
                        input_ids.append(self.sep_token_id)

                        for j in range(len(context)-1, -1, -1):
                            if j % 2 == 1:
                                max_length = 64 ## response_length
                            else:
                                max_length = 64 ## query_length

                            cur_utt = context[j]
                            if cur_utt == "": ## For TREC CAsT 2019
                                continue

                            if first_context:
                                cur_utt = "context: " + cur_utt
                                first_context = False

                            utt = self.tokenizer.encode(cur_utt, add_special_tokens=False, max_length=max_length, truncation=True)
                            if (len(input_ids) + len(utt) + 1) > self.max_concat_length:
                                input_ids += utt[:(self.max_concat_length - len(input_ids) - 1)] + [self.sep_token_id] # must ended with [SEP]
                                break
                            else:
                                utt = utt + [self.sep_token_id]
                                input_ids.extend(utt)

                        input_ids, _ = padding_seq_to_same_length(input_ids, self.max_concat_length, self.pad_token_id)
                        candidates_ids.append(input_ids)

                    # self.examples.append([sample_id, self.bert_pad(candidates_ids), ranks])
                    self.examples.append([sample_id, candidates_ids, ranks])
                else:
                    reformulation_text = []
                    all_rewrites = []
                    all_responses = []
                    context = record["context"]
                    query = record["current_query"]

                    for rewrite, response in zip(record["predicted_rewrite"], record["predicted_response"], strict=False):
                        # ids = self.bert_encode(rewrite, response)
                        reformulation = f"rewrite: {rewrite} {response}"
                        # reformulation = f"rewrite: {rewrite}"
                        # reformulation = f"question: {rewrite}"
                        query = f"query: {query}"
                        first_context = True

                        input_ids = []
                        tokenized_rewrite = self.tokenizer.encode(reformulation, add_special_tokens=True, max_length=self.max_query_len)
                        input_ids.extend(tokenized_rewrite)
                        # input_ids.append(self.sep_token_id)

                        tokenized_query = self.tokenizer.encode(query, add_special_tokens=False, max_length=self.max_query_len)
                        input_ids.extend(tokenized_query)
                        input_ids.append(self.sep_token_id)

                        for j in range(len(context)-1, -1, -1):
                            if j % 2 == 1:
                                max_length = 64 ## response_length
                            else:
                                max_length = 64 ## query_length

                            cur_utt = context[j]
                            if first_context:
                                cur_utt = "context: " + cur_utt
                                first_context = False

                            utt = self.tokenizer.encode(cur_utt, add_special_tokens=False, max_length=max_length, truncation=True)
                            if len(input_ids) + len(utt) > self.max_concat_length:
                                input_ids += utt[:self.max_concat_length - len(input_ids) - 1] + [self.sep_token_id] # must ended with [SEP]
                                break
                            else:
                                utt = utt + [self.sep_token_id]
                                input_ids.extend(utt)

                        input_ids, _ = padding_seq_to_same_length(input_ids, self.max_concat_length)

                        candidates_ids.append(input_ids)
                        reformulation_text.append(f"{rewrite} {response}")
                        all_rewrites.append(rewrite)
                        all_responses.append(response)

                    # self.examples.append([sample_id, self.bert_pad(candidates_ids), reformulation_text])
                    self.examples.append([sample_id, candidates_ids, reformulation_text, all_rewrites, all_responses])

            elif input_type == "contextual":
                raise NotImplementedError

    def bert_pad(self, X):
        max_len = self.max_concat_length
        if max_len < 0:
            max_len = max(len(x) for x in X)
        result = []
        for x in X:
            if len(x) < max_len:
                x.extend([self.pad_token_id] * (max_len - len(x)))
            result.append(x)
        return result

    def bert_encode(self, rewrite: str, response: str):
        rewrite_ids = self.tokenizer.encode(rewrite, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        ids = [self.cls_token_id]

        ids.extend(rewrite_ids[:self.args.max_len -len(ids) -3])
        ids.append(self.sep_token_id)
        ids.extend(response_ids[:self.args.max_len - len(ids) -3])
        ids.append(self.sep_token_id)

        return ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        def collate_fn(batch: list):
            if args.test is True:
                collated_dict = {
                    "sample_ids": [],
                    "candidates_ids": [],
                    "reformulation": [],
                    "rewrite": [],
                    "response": [],
                }
                for sample_id, candidates_ids, reformulation, rewrite, response in batch:
                    collated_dict["sample_ids"].append(sample_id)
                    collated_dict["candidates_ids"].append(candidates_ids)
                    collated_dict["reformulation"].append(reformulation)
                    collated_dict["rewrite"].append(rewrite)
                    collated_dict["response"].append(response)
            else:
                collated_dict = {
                    "sample_ids": [],
                    "candidates_ids": [],
                    "ranks": [],
                }
                for sample_id, candidates_ids, ranks in batch:
                    collated_dict["sample_ids"].append(sample_id)
                    collated_dict["candidates_ids"].append(candidates_ids)
                    collated_dict["ranks"].append(ranks)

            not_need_to_tensor_keys = ["sample_ids", "reformulation", "rewrite", "response"]
            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long) # type: ignore

            return collated_dict

        return collate_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    args.test = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # file_name = "./Projects/AdaRewriter/datasets/toys.jsonl"
    file_name = "./AdaRewriter/datasets/topiocqa_val_with_cos_ranks_full.json"

    dataset = AdapterDatasetWithContextV2(args, tokenizer, file_name, input_type="direct", is_test=args.test)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=AdapterDatasetWithContextV2.get_collate_fn(args))

    for batch in test_loader:
        for cand in batch["candidates_ids"]:
            print(tokenizer.decode(cand[0], skip_special_tokens=False))
        break

