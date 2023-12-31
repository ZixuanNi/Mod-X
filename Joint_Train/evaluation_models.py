from __future__ import print_function
import os
import pickle
import json
import torch
import numpy
from eval_data import get_split_loader
#from coco_eval import get_split_loader_coco
import time
import numpy as np
from collections import OrderedDict
import clip
from PIL import Image
from tqdm import tqdm
#from model import *

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def encode_data(args,model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    model.eval()
    print ("Evaluating...")

    img_embs = None
    cap_embs = None
    with torch.no_grad():

        for i, (images, captions, index, image_name) in tqdm(enumerate(data_loader)):
            batch_size = images.shape[0]
            captions = torch.cat([clip.tokenize(c) for c in captions])

            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()

            img_emb = model.encode_image(images)
            cap_emb = model.encode_text(captions)

            img_emb = img_emb / img_emb.norm(dim = -1,keepdim=True)
            cap_emb = cap_emb / cap_emb.norm(dim = -1,keepdim=True)

            if img_embs is None:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

            for idx in range(batch_size):
                img_embs[i * args.batch_size + idx] = img_emb.data.cpu().numpy().copy()[idx]
                cap_embs[i * args.batch_size + idx] = cap_emb.data.cpu().numpy().copy()[idx]

        del images, captions

    return img_embs, cap_embs

def evalrank(args,stats_file):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print ("Running on: ", device)

    model_clip, preprocess = clip.load(args.cnn, device=device,jit=False)

    print('Loading dataset')
    data_loader = get_split_loader(args.split, args.data_name, args.batch_size, args.workers, args, preprocess)

    print('Computing results...')
    if args.clip:
        img_embs, cap_embs = encode_data(args,model_clip, data_loader)
    else:

        model = Clip_Linear(model_clip, args)
        weights = torch.load(args.weights)['model']
        model.load_state_dict(weights)
        model.cuda()
        img_embs, cap_embs = encode_data(model, data_loader)

    if args.data_name == 'wiki':
        npts = 1
        caps_per_image = 2
    else:
        npts = None
        caps_per_image = 5
    
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0]/caps_per_image , cap_embs.shape[0]))


    r, rt = i2t(img_embs, cap_embs, return_ranks=True, npts=npts)
    ri, rti = t2i(img_embs, cap_embs, return_ranks=True, npts=npts)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

    stats = dict(I2T=r,T2I=ri)
    print(json.dumps(stats))
    print(json.dumps(stats), file=stats_file)


    

def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        caps_per_image = 5
    else:
        # Wiki
        caps_per_image = 2

    npts = images.shape[0] / caps_per_image

    index_list = []
    npts = int(npts)

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        im = images[caps_per_image * index].reshape(1, images.shape[1])

        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1] 
        index_list.append(inds[0]) 

        rank = 1e20
        for i in range(caps_per_image * index, caps_per_image * index + caps_per_image, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        caps_per_image = 5
    else:
        caps_per_image = 2

    npts = images.shape[0] / caps_per_image

    ims = numpy.array([images[i] for i in range(0, len(images), caps_per_image)])
    npts = int(npts)

    ranks = numpy.zeros(caps_per_image * npts)
    top1 = numpy.zeros(caps_per_image * npts)
    for index in range(npts):

        queries = captions[caps_per_image * index:caps_per_image * index + caps_per_image]

        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[caps_per_image * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[caps_per_image * index + i] = inds[i][0]

    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)