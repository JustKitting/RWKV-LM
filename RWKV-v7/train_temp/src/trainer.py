import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

from .eval_utils import EvaluationManager

def my_save(args, trainer, dd, ff):
    if 'deepspeed_stage_3' in args.strategy:
        trainer.save_checkpoint(ff, weights_only=True)
    else:
        torch.save(dd, ff)

class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.evaluator = EvaluationManager(args)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args

        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        monitor = getattr(args, "_warp_monitor", None)
        if monitor is not None:
            monitor.set_step(int(real_step))

        # LR schedule
        w_step = args.warmup_steps

        lr = args.lr_init

        if args.my_exit_tokens != 0: # cosine decay
            real_tokens = real_step * args.ctx_len * args.real_bsz
            warmup_tokens = w_step * args.ctx_len * args.real_bsz
            progress = (real_tokens - warmup_tokens) / (abs(args.my_exit_tokens) - warmup_tokens)
            progress = max(0, min(1, progress))
            lr_final_factor = args.lr_final / args.lr_init                
            lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
            if args.my_exit_tokens > 0:
                lr = args.lr_init * lr_mult
            else:
                lr = (lr + args.lr_init * lr_mult) / 2
            if progress >= 1:
                if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):
                    my_save(
                        args, trainer,
                        pl_module.state_dict(),
                        f"{args.proj_dir}/rwkv-final.pth",
                    )
                    exit(0)
        if trainer.global_step < w_step:
            lr = lr * (0.01 + 0.99 * trainer.global_step / w_step)

        wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            param_group["lr"] = lr * param_group["my_lr_scale"]

        trainer.my_lr = lr
        trainer.my_wd = wd_now

        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print("Login to wandb...")
                    # HERE set args.wandb to your personal project name before launching training
                    import wandb
                    if len(args.wandb_api_key) > 0:
                        wandb.login(key=args.wandb_api_key, relogin=True)  # HERE sourced from CLI to avoid manual env export
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now

            if hasattr(trainer, "my_loss_all"):
                loss_tensor = trainer.my_loss_all.float()
                trainer.my_loss = loss_tensor.mean().item()
            else:
                loss_value = outputs
                if isinstance(loss_value, dict) and "loss" in loss_value:
                    loss_value = loss_value["loss"]
                if torch.is_tensor(loss_value):
                    trainer.my_loss = loss_value.detach().float().mean().item()
                else:
                    trainer.my_loss = float(loss_value)
                trainer.my_loss_all = torch.tensor([trainer.my_loss], dtype=torch.float32, device=pl_module.device)

            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)

            if trainer.logger is not None:
                try:
                    tb = trainer.logger.experiment
                    if hasattr(tb, "add_scalar"):
                        tb.add_scalar("train/loss", trainer.my_epoch_loss, real_step)
                        tb.add_scalar("train/lr", trainer.my_lr, real_step)
                        if kt_s > 0:
                            tb.add_scalar("train/kt_per_s", kt_s, real_step)
                        if hasattr(tb, "flush"):
                            tb.flush()
                except Exception as log_err:
                    rank_zero_info(f"TensorBoard log failed: {log_err}")

            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                monitor = getattr(args, "_warp_monitor", None)
                if monitor is not None:
                    warp_metrics = monitor.pop_metrics(int(real_step))
                    lll.update(warp_metrics)
                trainer.my_wandb.log(lll, step=int(real_step))

        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy): # save pth
            if args.magic_prime > 0:
                if int(real_step) == int(args.magic_prime // args.real_bsz) - 1:
                    to_save_dict = pl_module.state_dict()
                    my_save(
                        args, trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-final.pth",
                    )

        monitor = getattr(args, "_warp_monitor", None)
        if monitor is not None:
            monitor.flush()


    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        container = trainer.train_dataloader.dataset
        dataset = getattr(container, "datasets", container)
        if hasattr(dataset, "set_distributed"):
            dataset.set_distributed(
                trainer.global_rank,
                trainer.world_size,
                int(args.epoch_begin + trainer.current_epoch),
            )
        else:
            dataset.global_rank = trainer.global_rank
            dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
            dataset.world_size = trainer.world_size
        # print(f'########## world_size {trainer.world_size} global_rank {trainer.global_rank} real_epoch {int(args.epoch_begin + trainer.current_epoch)} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        real_epoch = args.epoch_begin + trainer.current_epoch
        should_checkpoint = (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1)
        to_save_dict = {}
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if should_checkpoint:
                if args.data_type == 'wds_img':
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith('encoder.') or k.startswith('decoder.'):
                            to_save_dict[k] = raw_dict[k]
                else:
                    to_save_dict = pl_module.state_dict()
                try:
                    my_save(
                        args, trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth",
                    )
                except Exception as e:
                    print('Error\n\n', e, '\n\n')

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{real_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0

            eval_scalars = {}
            eval_results = []
            try:
                eval_scalars, eval_results = self.evaluator.run_epoch_end(
                    pl_module,
                    real_epoch=real_epoch,
                    global_step=int(trainer.global_step),
                    should_checkpoint=should_checkpoint,
                )
            except Exception as eval_exc:
                eval_results = [
                    {
                        "label": "eval_manager_error",
                        "scalars": {},
                        "details": {"error": str(eval_exc)},
                    }
                ]

            for res in eval_results:
                if isinstance(res, dict):
                    label = res.get("label", "eval")
                    scalars = res.get("scalars", {})
                    details = res.get("details", {})
                else:
                    label = res.label
                    scalars = res.scalars
                    details = res.details

                if scalars:
                    metrics_fmt = ", ".join(f"{k}={v:.6f}" for k, v in scalars.items())
                    trainer.my_log.write(f"# {label}: {metrics_fmt} | details={details}\n")
                else:
                    trainer.my_log.write(f"# {label}: details={details}\n")

            trainer.my_log.flush()

            if eval_scalars:
                if trainer.logger is not None:
                    try:
                        tb = trainer.logger.experiment
                        if hasattr(tb, "add_scalar"):
                            for name, value in eval_scalars.items():
                                tb.add_scalar(name, value, trainer.global_step)
                            if hasattr(tb, "flush"):
                                tb.flush()
                    except Exception as log_err:
                        rank_zero_info(f"TensorBoard eval log failed: {log_err}")

                if len(args.wandb) > 0:
                    try:
                        trainer.my_wandb.log(eval_scalars, step=int(trainer.global_step))
                    except Exception as wandb_err:
                        rank_zero_info(f"wandb eval log failed: {wandb_err}")

        monitor = getattr(args, "_warp_monitor", None)
        if monitor is not None:
            monitor.flush()

@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.train_stage == 1:
        if len(model.args.load_model) > 0:
            print(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except:
                    print('missing', k)
                    exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except:
                    tmp = mm[k].squeeze().clone()
                    print(k, src.shape, '-->', mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss-1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1-ii) + src[p0+1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    print(sss[:10], '...', sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    print(mmm[:10], '...', mmm[-10:])

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.train_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)
