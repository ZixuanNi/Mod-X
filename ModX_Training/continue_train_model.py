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
from data_loader.flickr30k_dataloader import  CLIP_flickr30k_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from data_loader.cifar100_dataloader import  CLIP_cifar100_dataset
from torch.utils.data import ConcatDataset
import json
from torch.utils.data import Subset

MODEL_CONFIG_PATH = 'model_config.yaml'

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint_path", 
    default=None, 
    type=str, required=True, 
    help="path of trained checkpoint"
)

parser.add_argument(
    "--save_steps", 
    default=1000, 
    type=int, 
    required=False, 
)

parser.add_argument(
    "--train_img_dir", 
    default=None, 
    type=Path, 
    required=False, 
)

parser.add_argument(
    "--train_annotation_file", 
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
    default=1e-08, 
    type=float, 
    required=False, 
)

parser.add_argument(
    "--train_dir",
    type=Path,
    default=None,
    help="For fine tuning or linear probe, which dataset to train on",
)

parser.add_argument(
    "--saved_checkpoints",
    type=Path,
    default=None,
)

parser.add_argument(
    "--start", 
    default=0, 
    type=int, 
    required=False, 
)

parser.add_argument(
    "--end", 
    default=0, 
    type=int, 
    required=False, 
)

parser.add_argument(
    "--alpha", 
    default=0.1, 
    type=float, 
    required=True, 
)

parser.add_argument(
    "--model_name", 
    default=RN, 
    type=str, 
    required=False, 
)

args = parser.parse_args()

args.train_dir.mkdir(parents=True, exist_ok=True)
stats_file = open(args.train_dir / 'stats.txt', 'a', buffering=1)
print(' '.join(sys.argv))
print(' '.join(sys.argv), file=stats_file)

def load_config_file(file_path):
    with open(file_path, 'r') as fp:
        return OmegaConf.load(fp)
    
model_config = load_config_file(MODEL_CONFIG_PATH)

states = dict(configs = model_config)
print(states, file=stats_file)

device = "cuda" if torch.cuda.is_available() else "cpu"
n_gpu = torch.cuda.device_count()

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

tokenizer = SimpleTokenizer()

################################# 模型导入 RN50或者ViT ##############################
if args.model_name == 'RN':
    model_params = dict(model_config.RN50)
    model_params['vision_layers'] = tuple(model_params['vision_layers'])
    model_params['vision_patch_size'] = None
    model = CLIP(**model_params)
    model_old = CLIP(**model_params)

    #################### 导入上一个模型 #######################
    checkpoint_old = torch.load(args.checkpoint_path)
    state_dict_old = checkpoint_old['model_state_dict']
    model_old.load_state_dict(state_dict_old)

    #################### 导入当前模型 #######################
    checkpoint = torch.load(args.checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)

else:
    model_params = dict(model_config.ViTB32)
    model = CLIP(**model_params)
    model_old = CLIP(**model_params)

    ####修改 vit 参数结构
    def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            if p.requires_grad:
                p.grad.data = p.grad.data.float()

    if args.start == 0 :
        ###首次训练 使用jit 导入模型checkpoint
        model_state = torch.jit.load(args.checkpoint_path, map_location="cpu").eval()
        state_dict = model_state.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]

        model.load_state_dict(state_dict)
        model_old.load_state_dict(state_dict)

    ### 若已经过finetune，直接load就可以
    else:
        checkpoint = torch.load(args.checkpoint_path)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        model_old.load_state_dict(state_dict)

    ### 将模型参数中的数据格式全部换位 float
    for p in model.parameters():
        p.data = p.data.float() 

print("Training/evaluation ...") 

#################### 认知对齐模块 #########################
def cognition_align(old,new):
    n,m = old.shape
    index = torch.max(old,1)[1] ### 找到对比矩阵中每一行的最大值索引
    for i in range(n):
        if index[i] != i: ### 如果最大值索引不是当前行的索引值，则说明过去模型对当前该样本数据认知错误，为此将这一行替换为当前训练模型的对应矩阵行，排除其对当前训练模型的影响。
            old[i] = new[i]

    return old,new


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


#train_coco_dataset=CLIP_COCO_dataset(args, tokenizer)
#train_cifar100_dataset=CLIP_cifar100_dataset(root='/home/nzx/ContinualLearning/datasets',args=args, text_tokenizer=tokenizer)
train_flickr30k_dataset=CLIP_flickr30k_dataset(args, tokenizer)
#concat_data = ConcatDataset([train_coco_dataset, train_cifar100_dataset])
index_list = [0,int(len(train_flickr30k_dataset)/5),int(len(train_flickr30k_dataset)/5*2),int(len(train_flickr30k_dataset)/5*3),int(len(train_flickr30k_dataset)/5*4),int(len(train_flickr30k_dataset)/5*5)]

index_train = [i for i in range(index_list[args.start],index_list[args.end])]
train_dataset = Subset(train_flickr30k_dataset,index_train)
concat_data = train_dataset

train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpu) 
train_loader = get_dataloader(args, concat_data, n_gpu)

t_total = len(train_loader) // args.gradient_accumulation_steps \
            * args.num_train_epochs

optimizer = AdamW(model.parameters(), lr=args.params_lr, eps=args.params_eps, weight_decay=args.params_weight_decay)
num_warmup_steps = int(0.20 * t_total)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= num_warmup_steps, num_training_steps= t_total)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    model_old = torch.nn.DataParallel(model_old) 


model = model.to(torch.device(device))
model.train()

model_old = model_old.to(torch.device(device))
model_old.eval() ### 限制旧模型的更新

global_step, global_loss, global_acc =0,  0.0, 0.0
model.zero_grad()

for epoch in range(int(args.num_train_epochs)):
    for step, batch in enumerate(train_loader):
        input_images, input_texts = batch

        input_images = input_images.to(torch.device(device))
        input_texts = input_texts.to(torch.device(device))
        
        ### 读取新旧模型在当前训练样本上的表征
        image_features, text_features = model(input_images, input_texts)
        image_features_old,text_features_old = model_old(input_images,input_texts)

        ### 对新旧模型当前样本表征进行归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features_old = image_features_old / image_features_old.norm(dim=-1, keepdim=True)
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)


        if n_gpu == 1:
            logit_scale = model.logit_scale.exp()
        elif n_gpu > 1:
            logit_scale = model.module.logit_scale.exp()

        ### 构造新旧模型在当前样本上的对比矩阵
        per_image = image_features @ text_features.t()
        per_text = text_features @ image_features.t()

        img_text_old = image_features_old @ text_features_old.t()
        text_img_old = text_features_old @ image_features_old.t()

        ### 套用 cognition_align 模块剔除错误认知信息 
        logits_img_text_old,logits_per_image = cognition_align(img_text_old,per_image)
        logits_text_img_old,logits_per_text = cognition_align(text_img_old,per_text)

        off_dia_per_image = logits_per_image
        off_dia_img_text_old = logits_img_text_old

        off_dia_per_text = logits_per_text
        off_dia_text_img_old = logits_text_img_old


        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

        ### 使用KL散度函数对非对角线信息分布进行对齐
        loss_dia_img = F.kl_div(off_dia_per_image.softmax(dim=-1).log(), off_dia_img_text_old.softmax(dim=-1), reduction='sum')
        loss_dia_text = F.kl_div(off_dia_per_text.softmax(dim=-1).log(), off_dia_text_img_old.softmax(dim=-1), reduction='sum')

        ### 标准训练当前模型的对比误差
        image_loss = F.cross_entropy(logit_scale * logits_per_image, labels)
        text_loss  = F.cross_entropy(logit_scale * logits_per_text, labels)

        loss_train = (image_loss + text_loss) / 2 
        loss_distill = (loss_dia_img + loss_dia_text) / 2

        ### 使用超参数 alpha 结合两个训练误差 
        all_loss = loss_train + args.alpha*loss_distill


        if n_gpu > 1: 
            all_loss = all_loss.mean()
        if args.gradient_accumulation_steps > 1:
            all_loss = all_loss / args.gradient_accumulation_steps

        all_loss.backward()
        global_loss += all_loss.item()

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
                stats = dict(Epoch=epoch,loss=all_loss.item())
                print(json.dumps(stats), file=stats_file)

            if global_step % 50 == 0:
                print("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f})".format(epoch, global_step, 
                    optimizer.param_groups[0]["lr"], all_loss.item(), global_loss / global_step)
                )

            if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                    global_step == t_total:
                save_checkpoint(args, epoch, global_step, model, optimizer,n_gpu) 

avg_loss = global_loss / global_step

print("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)













