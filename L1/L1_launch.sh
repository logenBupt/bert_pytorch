ulimit -n 100000
export NCCL_SOCKET_IFNAME=eth0

export BASE_DIR=/webdata-nfs/kwtang/bling/Embedding
NOW=$(date +"%Y%m%d_%H%M")
export UUID=$(cat /proc/sys/kernel/random/uuid)
export model_type=l1_100d_cls_shared
export JOBID=${NOW}_${model_type}_${UUID}
# export JOB_CODE_DIR=/job/${JOBID}/$(basename ${BASE_DIR})
export JOB_CODE_DIR=/webdata-nfs/kwtang/job/${JOBID}/$(basename ${BASE_DIR})
rm -rf ${JOB_CODE_DIR}
sudo mkdir -p ${JOB_CODE_DIR}
sudo chown `whoami` ${JOB_CODE_DIR}

scriptname=$(readlink -f "$0")
cp $scriptname ${JOB_CODE_DIR}
echo "Copying to ${JOB_CODE_DIR}"
cd ${BASE_DIR}
find . -type f -name "*.py" -exec cp --parents {} ${JOB_CODE_DIR} \;

echo "Job id: ${JOBID}"

cd ${JOB_CODE_DIR}/L1

# --optimizer lamb
# train_cmd="python3 -m torch.distributed.launch --nproc_per_node=8 driver.py --model_type ${model_type}  --model_name_or_path bert-base-multilingual-cased --task_name MSMarco --do_train --evaluate_during_training --data_dir /webdata-nfs/kwtang/L1_data  --per_gpu_eval_batch_size=1024  --per_gpu_train_batch_size=128   --data_cache_dir /webdata-nfs/kwtang/L1/cache/${JOBID}    --learning_rate 1e-4  --logging_steps 500   --num_train_epochs 3.0   --output_dir /mnt/azureblob/kwtang/L1/${JOBID}/ --warmup_steps 1000  --overwrite_output_dir  --gradient_accumulation_steps 1  --expected_train_size 60000000 --logging_steps_per_eval 20 --log_dir /home/kwtang/tensorboard/${DLWS_JOB_ID}/logs/${JOBID} --fp16 --optimizer lamb"

# train_cmd="python3 -m torch.distributed.launch --nproc_per_node=8 driver.py --model_type ${model_type}  --model_name_or_path bert-base-multilingual-cased --task_name MSMarco --do_train --evaluate_during_training --data_dir /webdata-nfs/kwtang/L1_data  --per_gpu_eval_batch_size=1024  --per_gpu_train_batch_size=256   --data_cache_dir /webdata-nfs/kwtang/L1/cache/${JOBID}    --learning_rate 8e-5  --logging_steps 500   --num_train_epochs 3.0   --output_dir /mnt/azureblob/kwtang/L1/${JOBID}/ --warmup_steps 1000  --overwrite_output_dir  --gradient_accumulation_steps 1  --expected_train_size 40000000 --logging_steps_per_eval 20 --log_dir /home/kwtang/tensorboard/${DLWS_JOB_ID}/logs/${JOBID} --fp16 --optimizer adamW"

# train_cmd="python3 -m torch.distributed.launch --nproc_per_node=8 driver.py --model_type ${model_type}  --model_name_or_path xlm-roberta-base --task_name MSMarco --do_train --evaluate_during_training --data_dir /webdata-nfs/kwtang/L1_data  --per_gpu_eval_batch_size=1024  --per_gpu_train_batch_size=256   --data_cache_dir /webdata-nfs/kwtang/L1/cache/${JOBID}    --learning_rate 8e-5  --logging_steps 500   --num_train_epochs 8.0   --output_dir /mnt/azureblob/kwtang/L1/${JOBID}/ --warmup_steps 1000  --overwrite_output_dir  --gradient_accumulation_steps 1  --expected_train_size 40000000 --logging_steps_per_eval 4 --log_dir /home/kwtang/tensorboard/${DLWS_JOB_ID}/logs/${JOBID} --fp16 --optimizer adamW"

train_cmd="python3 -m torch.distributed.launch --nproc_per_node=8 driver.py --model_type ${model_type}  --model_name_or_path /mnt/azureblob/kwtang/L1/20200611_0204_l1_100d_cls_shared_b4684cf7-8eea-4b37-bee9-77dbd770a62d/checkpoint-140000 --task_name MSMarco --do_eval --evaluate_during_training --data_dir /webdata-nfs/kwtang/L1_data  --per_gpu_eval_batch_size=1024  --per_gpu_train_batch_size=128   --data_cache_dir /webdata-nfs/kwtang/L1/cache/${JOBID}    --learning_rate 8e-5  --logging_steps 500   --num_train_epochs 3.0   --output_dir /mnt/azureblob/kwtang/L1/${JOBID}/ --warmup_steps 1000  --overwrite_output_dir  --gradient_accumulation_steps 1  --expected_train_size 40000000 --logging_steps_per_eval 20 --log_dir /home/kwtang/tensorboard/${DLWS_JOB_ID}/logs/${JOBID} --fp16 --optimizer adamW --eval_full"

echo $train_cmd
eval $train_cmd