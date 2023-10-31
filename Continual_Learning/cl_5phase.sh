echo "start train_step1"
CUDA_VISIBLE_DEVICES=2,3,4,5 python continual_train_model.py \
--checkpoint_path ./RN50_COCO/checkpoint_final.pt \
--train_img_dir /mnt/nzx/Flickr30K/flickr30k-images/ \
--train_annotation_file /mnt/nzx/Flickr30K/dataset_flickr30k.json \
--train_dir ./checkpoint_CL_bs70/phase_1/ \
--saved_checkpoints ./checkpoint_CL_bs70/phase_1/ \
--per_gpu_train_batch_size 70 --start 0 --end 1 --params_lr 5e-4
echo "end train_step1"
wait

echo "start train_step2"
CUDA_VISIBLE_DEVICES=2,3,4,5 python continual_train_model.py \
--checkpoint_path ./checkpoint_CL_bs70/phase_1/checkpoint_final.pt \
--train_img_dir /mnt/nzx/Flickr30K/flickr30k-images/ \
--train_annotation_file /mnt/nzx/Flickr30K/dataset_flickr30k.json \
--train_dir ./checkpoint_CL_bs70/phase_2/ \
--saved_checkpoints ./checkpoint_CL_bs70/phase_2/ \
--per_gpu_train_batch_size 70 --start 1 --end 2 --params_lr 5e-4
echo "end train_step2"
wait

echo "start train_step3"
CUDA_VISIBLE_DEVICES=2,3,4,5 python continual_train_model.py \
--checkpoint_path ./checkpoint_CL_bs70/phase_2/checkpoint_final.pt \
--train_img_dir /mnt/nzx/Flickr30K/flickr30k-images/ \
--train_annotation_file /mnt/nzx/Flickr30K/dataset_flickr30k.json \
--train_dir ./checkpoint_CL_bs70/phase_3/ \
--saved_checkpoints ./checkpoint_CL_bs70/phase_3/ \
--per_gpu_train_batch_size 70 --start 2 --end 3 --params_lr 5e-4
echo "end train_step3"
wait

echo "start train_step4"
CUDA_VISIBLE_DEVICES=2,3,4,5 python continual_train_model.py \
--checkpoint_path ./checkpoint_CL_bs70/phase_3/checkpoint_final.pt \
--train_img_dir /mnt/nzx/Flickr30K/flickr30k-images/ \
--train_annotation_file /mnt/nzx/Flickr30K/dataset_flickr30k.json \
--train_dir ./checkpoint_CL_bs70/phase_4/ \
--saved_checkpoints ./checkpoint_CL_bs70/phase_4/ \
--per_gpu_train_batch_size 70 --start 3 --end 4 --params_lr 5e-4
echo "end train_step4"
wait

echo "start train_step5"
CUDA_VISIBLE_DEVICES=2,3,4,5 python continual_train_model.py \
--checkpoint_path ./checkpoint_CL_bs70/phase_4/checkpoint_final.pt \
--train_img_dir /mnt/nzx/Flickr30K/flickr30k-images/ \
--train_annotation_file /mnt/nzx/Flickr30K/dataset_flickr30k.json \
--train_dir ./checkpoint_CL_bs70/phase_5/ \
--saved_checkpoints ./checkpoint_CL_bs70/phase_5/ \
--per_gpu_train_batch_size 70 --start 4 --end 5 --params_lr 5e-4
echo "end train_step5"
wait

