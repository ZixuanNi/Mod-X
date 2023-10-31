import math
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def save_checkpoint(args, epoch, global_step, model, optimizer,n_gpu):
    if epoch == args.num_train_epochs-1:
        checkpoint_path = os.path.join(args.saved_checkpoints, f'checkpoint_final.pt')
    else:
        checkpoint_path = os.path.join(args.saved_checkpoints, f'checkpoint_{epoch}_{global_step}.pt')
    save_num = 0
    while (save_num < 10):
        try:
            if n_gpu > 1:
                torch.save({
                    'epoch' : epoch,
                    'global_step' : global_step,
                    'model_state_dict' : model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
            else:
                torch.save({
                    'epoch' : epoch,
                    'global_step' : global_step,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)

            print("Save checkpoint to {}".format(checkpoint_path))
            break
        except:
            save_num += 1
    if save_num == 10:
        print("Failed to save checkpoint after 10 trails.")
    return






