import argparse
from dataclasses import dataclass

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from model import AdapterModel, RankingLoss
from utils.data_utils import AdapterDataset, AdapterDatasetWithContextV2

@dataclass
class TrainerArgs:
    project_name: str="AdaRewriter"
    model_name: str="microsoft/deberta-v3-base"
    lr: float = 1e-5
    warmup_ratio: float = 0.1
    max_len: int = 512
    test: bool = False
    wandb: bool = False
    train_path: str | None = "./Projects/AdaRewriter/datasets/toys.jsonl"
    valid_path: str | None = "./Projects/AdaRewriter/datasets/toys.jsonl"
    test_path: str | None = None
    train_batch_size: int = 2
    valid_batch_size: int = 2
    test_batch_size: int = 4
    default_root_dir: str = "./checkpoints"
    max_epochs: int = 10
    devices: int = 1
    gradient_accumulate: int = 1
    cand_num: int = 5
    debug: bool = False
    loss_type: str = "equal-divide"
    loss_margin: float = 1.0
    loss_alpha: float = 1.0
    checkpoint: str | None = None
    add_context: bool = False
    output_file: str = "./results/results.json"

class Adapter(pl.LightningModule):
    def __init__(self, args: TrainerArgs) -> None:
        super().__init__()
        self.args = args

        self.model = AdapterModel(args.model_name)
        if args.test:
            self.out = open(self.args.output_file, "w")

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(self.args.warmup_ratio*self.trainer.estimated_stepping_batches),
                                                    num_training_steps=int(self.trainer.estimated_stepping_batches))
        lr_scheduler = {
            'scheduler': scheduler,
            "interval": "step",
            'name': 'linear-lr'
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        all_candidates = batch["candidates_ids"].view(-1, self.args.max_len)
        all_scores = self.model(all_candidates)


        idx = 0
        for id, cur_rank in enumerate(batch["ranks"]):
            if id == 0:
                loss = self.args.loss_alpha * RankingLoss(
                    all_scores[idx:idx+self.args.cand_num], cur_rank,
                    margin=self.args.loss_margin, loss_type=self.args.loss_type, device=self.device)
            else:
                loss += self.args.loss_alpha * RankingLoss(
                    all_scores[idx:idx+self.args.cand_num], cur_rank,
                    margin=self.args.loss_margin, loss_type=self.args.loss_type, device=self.device)
            idx += self.args.cand_num

        log_dict = {"train/loss": loss.clone().detach().float()}

        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        all_candidates = batch["candidates_ids"].view(-1, self.args.max_len)
        all_scores: torch.Tensor = self.model(all_candidates)

        idx, cnt, acc, best_rank, pred_rank = 0, 0, 0, 0, 0

        for cur_rank in batch["ranks"]:
            cur_score = all_scores[idx:idx+self.args.cand_num]
            top1 = torch.argmin(cur_score, dim=0)
            top1_gt = int(torch.argmin(cur_rank, dim=0))
            min_rank = torch.min(cur_rank, dim=0)[0]
            top1_ranks = (min_rank == cur_rank).nonzero().squeeze(1)
            best_rank += int(cur_rank[int(top1_gt)])
            pred_rank += int(cur_rank[int(top1)])
            # pred_rank_list.append(int(cur_rank[int(top1)]))
            # top1s.append(int(top1))
            # top1s_gt.append(int(top1_gt))
            if top1 in top1_ranks:
                acc += 1
            cnt += 1
            idx += self.args.cand_num

        log_dict = {
            "eval/acc": acc / cnt if cnt > 0 else 0,
            "eval/best_rank": best_rank / cnt if cnt > 0 else 0,
            "eval/pred_rank": pred_rank / cnt if cnt > 0 else 0
        }

        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        all_candidates = batch["candidates_ids"].view(-1, self.args.max_len)
        all_scores: torch.Tensor = self.model(all_candidates)
        idx = 0

        for sample_id, reformulations in zip(batch["sample_ids"], batch["reformulation"], strict=False):
            cur_score = all_scores[idx:idx+self.args.cand_num]
            top1 = torch.argmin(cur_score, dim=0)
            # top1 = torch.argmax(cur_score, dim=0)
            final_reformulations = reformulations[int(top1)]
            
            self.out.write(f"{sample_id}\t{final_reformulations}\n")
            self.out.flush()
            idx += self.args.cand_num

def get_args() -> TrainerArgs:
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_len", type=int, default=TrainerArgs.max_len)
    parser.add_argument("--model_name", type=str, default=TrainerArgs.model_name)
    parser.add_argument("--test", action="store_true", default=TrainerArgs.test)
    parser.add_argument("--wandb", action="store_true", default=TrainerArgs.wandb)
    parser.add_argument("--train_path", type=str, default=TrainerArgs.train_path)
    parser.add_argument("--valid_path", type=str, default=TrainerArgs.valid_path)
    parser.add_argument("--test_path", type=str, default=TrainerArgs.test_path)
    parser.add_argument("--train_batch_size", type=int, default=TrainerArgs.train_batch_size)
    parser.add_argument("--valid_batch_size", type=int, default=TrainerArgs.valid_batch_size)
    parser.add_argument("--test_batch_size", type=int, default=TrainerArgs.test_batch_size)
    parser.add_argument("--default_root_dir", type=str, default=TrainerArgs.default_root_dir)
    parser.add_argument("--max_epochs", type=int, default=TrainerArgs.max_epochs)
    parser.add_argument("--devices", type=int, default=TrainerArgs.devices)
    parser.add_argument("--gradient_accumulate", type=int, default=TrainerArgs.gradient_accumulate)
    parser.add_argument("--debug", action="store_true", default=TrainerArgs.debug)
    parser.add_argument("--cand_num", type=int, default=TrainerArgs.cand_num)
    parser.add_argument("--checkpoint", type=str, default=TrainerArgs.checkpoint)
    parser.add_argument("--loss_margin", type=float, default=TrainerArgs.loss_margin)
    parser.add_argument("--loss_type", choices=["weight-divide", "equal-divide", "equal-sum"], default=TrainerArgs.loss_type)
    parser.add_argument("--add_context", action="store_true", default=TrainerArgs.add_context)
    parser.add_argument("--lr", type=float, default=TrainerArgs.lr)
    parser.add_argument("--output_file", type=str, default=TrainerArgs.output_file)
    return TrainerArgs(**vars(parser.parse_args()))


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not args.test:
        if args.train_path is None or args.valid_path is None:
            raise ValueError("Training file path is not provided.")

        logger = WandbLogger(project=args.project_name) if args.wandb else None
        run_name = logger.experiment.name if logger is not None else "debug"
        print(f"Run name: {run_name}")
        lr_monitor = LearningRateMonitor(logging_interval="step")

        if args.add_context:
            train_dataset = AdapterDatasetWithContextV2(args, tokenizer, args.train_path, input_type="direct", is_test=False)
        else:
            train_dataset = AdapterDataset(args, tokenizer, args.train_path, input_type="direct", is_test=False)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=train_dataset.get_collate_fn(args))

        # Uncomment to use validation dataset
        # if args.add_context:
        #     valid_dataset = AdapterDatasetWithContextV2(args, tokenizer, args.valid_path
        # else:
        #     valid_dataset = AdapterDataset(args, tokenizer, args.valid_path, input_type="direct", is_test=False)
        # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch

        model = Adapter(args)

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{args.default_root_dir}/{run_name}",
            save_top_k=10,
            save_weights_only=False,
            save_last=True,
            monitor="eval/pred_rank",
            save_on_train_epoch_end=True,
            enable_version_counter=False,
            verbose=True,
            mode="min"
        )

        trainer = pl.Trainer(
            logger=logger,
            devices=args.devices,
            precision="bf16-mixed",
            max_epochs=args.max_epochs,
            default_root_dir=args.default_root_dir,
            accumulate_grad_batches=args.gradient_accumulate,
            log_every_n_steps=10,
            # val_check_interval=0.05,
            fast_dev_run=args.debug,
            callbacks=[lr_monitor, checkpoint_callback]
        )
        
        ckpt_path = args.checkpoint
        if ckpt_path:
            print(f"[INFO] Resuming training from checkpoint: {ckpt_path}")
        else:
            print("[INFO] Starting training from scratch.")

        trainer.fit(
            model=model, 
            train_dataloaders=train_loader, 
            ckpt_path=ckpt_path
        )

    else:
        if args.test_path is None:
            raise ValueError("Test file path (--test_path) must be provided in --test mode.")

        if args.add_context:
            print("[INFO] Using dataset with context.")
            test_dataset = AdapterDatasetWithContextV2(args, tokenizer, args.test_path, input_type="direct", is_test=True)
            collate_fn = AdapterDatasetWithContextV2.get_collate_fn(args)
        else:
            print("[INFO] Using dataset without context.")
            test_dataset = AdapterDataset(args, tokenizer, args.test_path, input_type="direct", is_test=True)
            collate_fn = AdapterDataset.get_collate_fn(args)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn)
        
        model = Adapter(args)
        
        if args.checkpoint:
            print(f"[INFO] Loading fine-tuned checkpoint from: {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
        else:
            print(f"[INFO] No checkpoint provided. Using pre-trained model '{args.model_name}' directly.")
            
        model.eval()

        trainer = pl.Trainer(devices=args.devices)
        trainer.test(model, test_loader)
