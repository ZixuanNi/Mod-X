# dataloader here
from torch.utils.data import Dataset

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from omegaconf import OmegaConf
import os.path as op
import random
import torch
import numpy as np
from pathlib import Path
import json
from collections import OrderedDict

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), 
    ])

def get_img_id_to_img_path(annotations):
    img_id_to_img_path = {}
    for img_info in annotations['images']:
        if img_info['split'] == 'train' or img_info['split'] == 'val':
            img_id = img_info['imgid']
            file_name = img_info['filename']
            img_id_to_img_path[img_id] = file_name
    
    return img_id_to_img_path

def get_img_id_to_captions(annotations):
    img_id_to_captions = {}
    for img_info in annotations['images']:
        if img_info['split'] == 'train' or img_info['split'] == 'val':
            img_id = img_info['sentences'][0]['imgid']
            if img_id not in img_id_to_captions:
                img_id_to_captions[img_id] = []
            
            for i in range(len(img_info['sentences'])):
                caption = img_info['sentences'][i]['raw']
                img_id_to_captions[img_id].append(caption)
    
    return img_id_to_captions

class CLIP_flickr30k_dataset(Dataset):
    def __init__(self, args, text_tokenizer, context_length=77, input_resolution=224):
        super(CLIP_flickr30k_dataset, self).__init__()

        self.args = args

        annotation_file = self.args.train_annotation_file
        # print("annotation_file : ", annotation_file)
        annotations = read_json(annotation_file)

        self.img_id_to_filename = get_img_id_to_img_path(annotations)
        # print("img_id_to_filename : ", self.img_id_to_filename)

        self.img_id_to_captions = get_img_id_to_captions(annotations)

        self.img_ids = list(self.img_id_to_filename.keys())
        # print("total image ids = ", len(self.img_ids))

        self.img_dir = self.args.train_img_dir
        # print("img dir : ", self.img_dir)

        self.transform = _transform(input_resolution)
        self._tokenizer = text_tokenizer
        self.context_length = context_length


    def tokenize(self, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.context_length, dtype=torch.long)
        if len(tokens) <= self.context_length:
            result[:len(tokens)] = torch.tensor(tokens)
        else:
            result[:self.context_length] = torch.tensor(tokens)[:self.context_length]
        return result

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # randomly pick one caption from the image captions
        text = random.choice(self.img_id_to_captions[img_id])

        img_filename = self.img_id_to_filename[img_id]

        img_path = op.join(self.img_dir, img_filename)
        img = Image.open(img_path)
        img_input = self.transform(img)
        text_input = self.tokenize(text)

        return img_input, text_input




