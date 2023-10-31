# Mod-X
The reproduce of paper "Continual Vision-Language Representation Learning with Off-Diagonal Information" **ICML 2023**.

%We will release our code to the public as soon as possible after consulting with our collaborators.

After communicating with collaborators, I have open-sourced the experimental code in this paper, which is based on the COCO and Flickr30K datasets, as an individual. And this code has not been adapted for multi-node large-scale experiments.

This repository includes the following four sections:

* traditional continual finetuning: Continual_Training
* Joint train with all data/ traditional pretraining：Joint_train 
* Model's linear evaluation based on the DDP: linear_evaluation_ddp  
* Continual learning based on Mod-X：ModX_Training  

## Training:
### Joint training:
```python
cd ./Joint_Train

CUDA_VISIBLE_DEVICES=$gpu_id$ python pre_train_model.py 
--coco_train_img_dir $coco_img_path$ \
--coco_train_annotation_file $coco_caption_path$ \
--flickr30k_train_img_dir $f30k_img_path$ \
--flickr30k_train_annotation_file $f30k_caption_path$ \
--num_train_epochs 35 --train_dir $path_log$ \
--saved_checkpoints $save_checkpoint_path$ \
--params_weight_decay 1e-2 \
--params_eps 1e-5 \
--params_lr 5e-4 \
--per_gpu_train_batch_size 70

Alternatively, you can also directly use 'bash train.sh' within the 'Joint_Train' directory.
```
### Continual training:     
Using Joint_Training to obtain a trained/pretrained model on a specific dataset and save it to the path: /path/checkpoint/.
```python
cd ./Continual_Training
bash cl_5phase.sh
And - `--checkpoint_path`  is old model's checkpoint path
```  
If you want to train ViT, you just need to add args:
```python
--model_name ViT
```

### Continual Mod-X training:
Using Joint_Training to obtain a trained/pretrained model on a specific dataset and save it to the path: /path/checkpoint/.
```python
cd ./ModX_Training
bash modx_5phase.sh
And - `--checkpoint_path`  is old model's checkpoint path
```  
If you want to train ViT, you just need to add args:
```python
--model_name ViT
```
## Evaluating:
If you want to evaluate the linear separability of the CLIP model's visual encoder representations, you can 
```python
cd ./linear_evaluation_ddp
CUDA_VISIBLE_DEVICES=$gpu_id$ python evaluate_clip.py \
--epochs 100 \
--batch-size 256 \
--lr-classifier 0.3 \
--weight-decay 1e-6 \
--print-freq 100 \
--checkpoint-dir $/path/log/$
--cnn $/path/checkpoint$ \ 
--train_dir $/path/train_imagenet/$ \ %ImageNet PATH
--val_dir $path/val_imagenet/$ %ImageNet PATH
```  







