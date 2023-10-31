import torch

import evaluation_models
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='coco', help="{coco, f30k, wiki}")
    parser.add_argument('--cnn', default='ViT-B/16', help='[RN50, RN101, RN50x4, RN50x16, ViT-B/32, ViT-B/16]')
    parser.add_argument('--split', default='test', help='Split, test(1K), testall(5k) (Coco only)')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--clip', action='store_true', help='Evaluate with original CLIP model.')
    parser.add_argument('--weights', default='./runs/clip_ft_coco/model_best.pth.tar', help='Path of model to evaluate')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')

    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )

    print ("Simple Evaluator of CLIP on Image-Text Matching Datasets \n")
    args = parser.parse_args()
    print ("Arguments: ", args)

    stats_file = open(args.eval_dir / 'eval_stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)

    evaluation_models.evalrank(args,stats_file)

if __name__ == "__main__":
    main()
