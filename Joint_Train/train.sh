CUDA_VISIBLE_DEVICES=2,3,4,5 python pre_train_model.py --coco_train_img_dir /PATH/COCO/train2014/ \
--coco_train_annotation_file /PATH/COCO/annotations/captions_train2014.json \
--flickr30k_train_img_dir /PATH/Flickr30K/flickr30k-images --flickr30k_train_annotation_file /PATH/Flickr30K/dataset_flickr30k.json \
--num_train_epochs 35 --train_dir /PATH/checkpoint/coco+f30k/ \
--saved_checkpoints /PATH/coco+f30k/ --params_weight_decay 1e-2 --params_eps 1e-5 \
--params_lr 5e-4 --per_gpu_train_batch_size 70  #--gradient_accumulation_steps 10 








