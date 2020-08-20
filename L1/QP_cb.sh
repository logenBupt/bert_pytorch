ulimit -n 100000
export NCCL_SOCKET_IFNAME=eth0

export BASE_DIR=/home/chlaksh/bling/Embedding
NOW=$(date +"%Y%m%d_%H%M")
export UUID=$(cat /proc/sys/kernel/random/uuid)
export model_type=l1_qp_cb
export JOBID=${NOW}_${model_type}_${UUID}
export JOB_CODE_DIR=/webdata-nfs/chlaksh/job/${JOBID}/$(basename ${BASE_DIR})
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
train_cmd="python3 -m torch.distributed.launch --nproc_per_node=8 driver_cb.py --model_type ${model_type}  --model_name_or_path /webdata-nfs/chlaksh/QP/ --task_name QPwithCB --do_train --evaluate_during_training --data_dir /webdata-nfs/kwtang/L1  --per_gpu_eval_batch_size=512     --per_gpu_train_batch_size=32   --data_cache_dir /mnt/chlaksh/L1/cache/${JOBID}    --learning_rate 1e-5  --logging_steps 1000   --num_train_epochs 3.0   --output_dir /mnt/chlaksh/L1/${JOBID}/ --warmup_steps 1000  --overwrite_output_dir  --gradient_accumulation_steps 1  --expected_train_size 60000000 --logging_steps_per_eval 10 --log_dir /home/chlaksh/tensorboard/${DLWS_JOB_ID}/logs/${JOBID} --fp16 --optimizer lamb"

echo $train_cmd
eval $train_cmd
