ulimit -n 100000
export NCCL_SOCKET_IFNAME=eth0

export BASE_DIR=/webdata-nfs/kwtang/bling/Embedding
NOW=$(date +"%Y%m%d_%H%M")
export UUID=$(cat /proc/sys/kernel/random/uuid)
export model_type=l1_orig_100d_ann_bce
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

train_cmd="\
python3 -m torch.distributed.launch --nproc_per_node=8 ann_driver.py --model_type ${model_type} \
--model_name_or_path /mnt/azureblob/kwtang/L1/20200610_0627_l1_original_100d_44b42059-70c5-43ac-9747-a46a1afe789d/checkpoint-150000 --per_gpu_eval_batch_size=1024 --task_name MSMarco --preproc_dir /webdata-nfs/kwtang/L1_preproc/ --data_cache_dir /webdata-nfs/kwtang/L1/cache/${JOBID} \
--ann_dir /webdata-nfs/kwtang/L1_preproc/ann_data --per_gpu_train_batch_size=128 --data_dir /webdata-nfs/kwtang/L1_data \
--gradient_accumulation_steps 1 --learning_rate 1e-6 --output_dir /mnt/azureblob/kwtang/L1/${JOBID}/ \
--warmup_steps 10000 --logging_steps 500 --logging_steps_per_eval 10 --log_dir /home/kwtang/tensorboard/${DLWS_JOB_ID}/logs/${JOBID} --fp16 --optimizer adamW \
"

echo $train_cmd
eval $train_cmd