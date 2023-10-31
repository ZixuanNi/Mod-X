import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
from omegaconf import OmegaConf
from pathlib import Path
import sys
import random
from clip.model import CLIP
from torch.optim import Adam, AdamW
from utils import get_cosine_schedule_with_warmup,save_checkpoint
from clip.simple_tokenizer import SimpleTokenizer
from data_loader.coco_dataloader import  CLIP_COCO_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from data_loader.flickr30k_dataloader import  CLIP_flickr30k_dataset
from data_loader.RSICD_dataloader import CLIP_RSICD_dataset
from torch.utils.data import ConcatDataset
import json

MODEL_CONFIG_PATH = 'model_config.yaml'

parser = argparse.ArgumentParser()

parser.add_argument(
    "--save_steps", 
    default=1000, 
    type=int, 
    required=False, 
    help="save checkpoint every 1000 steps ",
)

parser.add_argument(
    "--coco_train_img_dir", 
    default=None, 
    type=Path, 
    required=False, 
)

parser.add_argument(
    "--coco_train_annotation_file", 
    default=None, 
    type=str, 
    required=False, 
)

parser.add_argument(
    "--flickr30k_train_img_dir", 
    default=None, 
    type=Path, 
    required=False, 
)

parser.add_argument(
    "--flickr30k_train_annotation_file", 
    default=None, 
    type=str, 
    required=False, 
)

parser.add_argument(
    "--RSICD_img_dir", 
    default=None, 
    type=Path, 
    required=False, 
)

parser.add_argument(
    "--RSICD_train_annotation_file", 
    default=None, 
    type=str, 
    required=False, 
)

parser.add_argument(
    "--num_train_epochs", 
    default=35, 
    type=int, 
    required=False, 
)

parser.add_argument(
    "--per_gpu_train_batch_size", 
    default=64, 
    type=int, 
    required=False, 
)

parser.add_argument(
    "--gradient_accumulation_steps", 
    default=1, 
    type=int, 
    required=False, 
)

parser.add_argument(
    "--params_lr", 
    default=5e-4, 
    type=float, 
    required=False, 
)

parser.add_argument(
    "--num_workers", 
    default=8, 
    type=int, 
    required=False, 
)

parser.add_argument(
    "--params_weight_decay", 
    default=0.1, 
    type=float, 
    required=False, 
)

parser.add_argument(
    "--params_eps", 
    default=1e-8, 
    type=float, 
    required=False, 
)

parser.add_argument(
    "--train_dir",
    type=Path,
    default=None,
    help="The path to save the training state",
)

parser.add_argument(
    "--saved_checkpoints",
    type=Path,
    default=None,
    help="The path to save the checkpoints",
)

parser.add_argument(
    "--model_name", 
    default=RN, 
    type=str, 
    required=False, 
)

args = parser.parse_args()

################################## The path to save the training state #############################
args.train_dir.mkdir(parents=True, exist_ok=True)
stats_file = open(args.train_dir / 'stats.txt', 'a', buffering=1)
print(' '.join(sys.argv))
print(' '.join(sys.argv), file=stats_file)
####################################################################################################

def load_config_file(file_path):
    with open(file_path, 'r') as fp:
        return OmegaConf.load(fp)

model_config = load_config_file(MODEL_CONFIG_PATH)
states = dict(configs = model_config)
print(states, file=stats_file)

device = "cuda" if torch.cuda.is_available() else "cpu"
n_gpu = torch.cuda.device_count()

seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

tokenizer = SimpleTokenizer()
###############################################################

if args.model_name == 'RN':
    model_params = dict(model_config.RN50)
    model_params['vision_layers'] = tuple(model_params['vision_layers'])
    model_params['vision_patch_size'] = None
    model = CLIP(**model_params)

else:
    model_params = dict(model_config.ViTB32)
    model = CLIP(**model_params)

print("Training/evaluation ...")



def get_dataloader(args, dataset, n_gpu, is_train = True):
    if is_train:
        sampler = RandomSampler(dataset)
        batch_size = args.per_gpu_train_batch_size * max(1, n_gpu)
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.per_gpu_eval_batch_size * max(1, n_gpu)

    dataloader = DataLoader(dataset, sampler=sampler, 
            batch_size=batch_size, num_workers=args.num_workers,drop_last=True)

    return dataloader


################################## load dataset ###############################################
train_coco_dataset=CLIP_COCO_dataset(args, tokenizer) 
train_flickr30k_dataset=CLIP_flickr30k_dataset(args, tokenizer) 
#train_RSICD = CLIP_RSICD_dataset(args,tokenizer)
concat_data = ConcatDataset([train_coco_dataset, train_flickr30k_dataset]) 
train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpu) 
train_loader = get_dataloader(args, concat_data, n_gpu, is_train=True)
##############################################################################################

t_total = len(train_loader) // args.gradient_accumulation_steps \
            * args.num_train_epochs

optimizer = AdamW(model.parameters(), lr=args.params_lr, eps=args.params_eps, weight_decay=args.params_weight_decay)
num_warmup_steps = int(0.20 * t_total)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= num_warmup_steps, num_training_steps= t_total)


if n_gpu > 1:
    model = torch.nn.DataParallel(model)

model = model.to(torch.device(device))
model.train()

global_step, global_loss, global_acc =0,  0.0, 0.0
model.zero_grad()

for epoch in range(int(args.num_train_epochs)):
    for step, batch in enumerate(train_loader):
        input_images, input_texts = batch

        input_images = input_images.to(torch.device(device))
        input_texts = input_texts.to(torch.device(device))

        image_features, text_features = model(input_images, input_texts) 

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if n_gpu == 1:
            logit_scale = model.logit_scale.exp()
        elif n_gpu > 1:
            logit_scale = model.module.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss  = F.cross_entropy(logits_per_text, labels)

        loss = (image_loss + text_loss) / 2

        if n_gpu > 1: 
            loss = loss.mean() 
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        global_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            global_step += 1
            optimizer.step()
            if n_gpu == 1:
                model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            elif n_gpu > 1:
                model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

            if scheduler:
                scheduler.step() 

            model.zero_grad()
            if epoch%5 == 0:
                stats = dict(Epoch=epoch,loss=loss.item())
                print(json.dumps(stats), file=stats_file)

            if global_step % 50 == 0:
                print("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f})".format(epoch, global_step, 
                    optimizer.param_groups[0]["lr"], loss.item(), global_loss / global_step)
                )

            if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                    global_step == t_total:
                save_checkpoint(args, epoch, global_step, model, optimizer,n_gpu) 
            

#return global_step, global_loss / global_step
avg_loss = global_loss / global_step

print("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)













