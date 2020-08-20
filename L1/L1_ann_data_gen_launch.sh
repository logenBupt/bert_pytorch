cmd="python3 -m torch.distributed.launch --nproc_per_node=8 L1_run_ann_data_gen.py --training_dir /mnt/azureblob/kwtang/L1/20200612_0354_l1_orig_100d_ann_bce_59338d6e-570f-44b2-87d2-88b2d833cbea/ --init_model_dir /mnt/azureblob/kwtang/L1/20200610_0627_l1_original_100d_44b42059-70c5-43ac-9747-a46a1afe789d/checkpoint-150000 \
--model_type l1_original_100d --model_name_or_path bert-base-multilingual-cased --output_dir /webdata-nfs/kwtang/L1_preproc/ann_data \
--data_dir /webdata-nfs/kwtang/L1_preproc --per_gpu_eval_batch_size 512 --topk_training 50 --negative_sample 3"

echo $cmd
eval $cmd