import torch
from torch import nn
from typing import Union
from transformers import AutoModel, AutoTokenizer

def RankingLoss(score, ranks, device, margin=1.0, loss_type="equal-divide", sorted=False):
    # ones = torch.ones_like(score)
    # loss_func = torch.nn.MarginRankingLoss(0.0)
    indices = torch.argsort(ranks)
    ranks = ranks[indices]
    score = score[indices]
    TotalLoss: torch.Tensor = torch.tensor(0.0).to(device) #loss_func(score, score, ones)
    # candidate loss
    n = score.size(0)
    total = 0
    for i in range(1, n):
        pos_score = score[:-i]
        neg_score = score[i:]
        rank_diff = ranks[i:] - ranks[:-i]
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        loss = torch.max(pos_score - neg_score + margin * rank_diff, torch.zeros_like(pos_score))
        if loss_type in ["equal-divide", "equal-sum"]:
            TotalLoss += torch.sum(loss)
            total += loss.size(0)
        elif loss_type in ["weight-divide", "weight-sum"]:
            TotalLoss += torch.mean(loss)
            total += 1
        else:
            raise NotImplementedError
    if loss_type in ["equal-divide", "weight-divide"] and total > 0: # Avoid division by zero
        TotalLoss /= total
    return TotalLoss

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = self.dense1(features)
        return x

class AdapterModel(nn.Module):
    def __init__(self, model_name: str="microsoft/deberta-v3-base") -> None:
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_token_id = tokenizer.pad_token_id

        self.model = AutoModel.from_pretrained(model_name)
        self.fc = MLPLayer(self.model.config)

    def forward(self, input_ids: torch.Tensor, attention_mask: Union[torch.Tensor, None] = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids != self.pad_token_id

        outputs = self.model(input_ids, attention_mask=attention_mask).last_hidden_state

        embddings = outputs[:, 0, :]
        scores = self.fc(embddings).squeeze(-1)

        return scores