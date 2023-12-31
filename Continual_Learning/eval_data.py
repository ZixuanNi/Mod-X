import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import json


class GenericDataset(data.Dataset):
    ### path of coco and f30k。
    def __init__(self, transform, split, data_name):
        self.transform = transform
        self.split = split
        self.data_name = data_name

        if self.data_name == 'coco':
            self.img_path = #'/PATH/COCO/'

        elif self.data_name =='f30k':
            self.img_path = #'/PATH/flickr30k-images/'
        else:
            self.img_path = #'/PATH/all_images/'
        
        with open ("./data_index/{}_{}_ims.txt".format(self.data_name, self.split), "r") as fp:
            self.images = fp.readlines()
        self.images = [image.strip() for image in self.images]
        
        with open ("./data_index/{}_{}_caps.txt".format(self.data_name, self.split), "r") as fp:
            self.captions = fp.readlines()
        self.captions = [caption.strip() for caption in self.captions]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        image_name = self.images[index]
        caption = self.captions[index]

        if self.data_name =='wiki':
            max_context_len = 20
        else:
            max_context_len = 60

        if self.data_name == 'f30k' or self.data_name == 'wiki':
            img_path = self.img_path
        else:
            coco_split_folder = image_name.split("_")[1]
            img_path = self.img_path + coco_split_folder + '/'

        short_caption = caption.split(' ')
        short_caption = short_caption[:max_context_len]
        caption = ' '.join(short_caption)

        image = self.transform(Image.open(img_path + image_name))

        return image, caption, index, image_name

    def __len__(self):
        if self.split == 'test' and self.data_name == 'coco': 
            return int(len(self.images))
        else:
            return len(self.images)

    def __len_text__(self):
        return int(len(self.captions))

def get_loader(transform, split, data_name, batch_size, num_workers, args,):
    """Returns torch.utils.data.DataLoader for custom dataset."""

    dataset = GenericDataset(transform, split, data_name)

    if split =='train':
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers)

    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers)
    return data_loader

def get_split_loader(split, data_name, batch_size, workers, args, preprocess):

    transform = preprocess

    loader = get_loader(transform, split, data_name, batch_size, workers, args,)

    return loader

