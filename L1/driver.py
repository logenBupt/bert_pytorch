""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

from os import listdir
from os.path import isfile, join
import sys
sys.path.append("..")

import argparse
import glob
import json
import logging
import os, shutil
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch import nn

# removed unnecessary clutter - import everything from lee
# from pairwise_model import MODEL_CLASSES, ALL_MODELS
from iterable_dataset import CachedStreamingDataLoader, SimplifiedStreamingDataset
from lamb import Lamb
from L1.process_fn import L1_process_fn, L1_process_fn_exp
from L1.config import L1ConfigDict, wrapped_process_fn, load_model_config
from L1.L1_eval import eval_fidelity
from util import LineShuffler

from datetime import datetime
from time import time
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    # new imports
    RobertaModel,
)

import copy
import csv
from torch import nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from sklearn.metrics import roc_curve, auc
import pandas as pd



def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj

def set_logger(log_path, args):
    logger = logging.getLogger('driver.py')
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%m/%d/%Y %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# --------------------------------------------------------------------------
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def save_checkpoint(args, global_step, model, tokenizer, optimizer=None, scheduler=None, logger=None):
    # Save model checkpoint
    if global_step<0:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception:
        pass

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    if logger:
        logger.info("Saving model checkpoint to %s", output_dir)
    else:
        print("Saving model checkpoint to %s", output_dir)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))


def train(args, model, tokenizer, shuffled_fh, train_fn, configObj, logger):
    """ Train the model """
    #if args.local_rank in [-1, 0]:
    tb_writer = None
    if is_first_worker():
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    real_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    total_train_steps = len(shuffled_fh) * args.num_train_epochs // real_batch_size
    if args.warmup_steps <= 0:
        args.warmup_steps = int(total_train_steps * args.warmup_proportion)

    if args.max_steps > 0:
        t_total = args.max_steps
        #args.num_train_epochs = args.max_steps // (args.expected_train_size // args.gradient_accumulation_steps) + 1 
    else:
        # t_total = args.expected_train_size // real_batch_size * args.num_train_epochs    
        t_total = total_train_steps
        args.max_steps = total_train_steps

    # layerwise optimization for lamb
    optimizer_grouped_parameters = []
    no_decay = ["bias", "LayerNorm.weight", "layer_norm", "LayerNorm"]
    layer_optim_params = set()
    for layer_name in ["bert.embeddings", "score_out", "downsample1", "downsample2", "downsample3", "embeddingHead"]:
         layer = getattr_recursive(model, layer_name)
         if layer is not None:
            optimizer_grouped_parameters.append({"params": layer.parameters()})
            for p in layer.parameters():
                layer_optim_params.add(p)

    if getattr_recursive(model, "bert.encoder.layer") is not None:
        for layer in model.bert.encoder.layer:
            optimizer_grouped_parameters.append({"params": layer.parameters()})
            for p in layer.parameters():
                layer_optim_params.add(p)
    optimizer_grouped_parameters.append({"params": [p for p in model.parameters() if p not in layer_optim_params]})
    
    if len(optimizer_grouped_parameters)==0:
        
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
    logger.info("len(optimizer_grouped_parameters): {}".format(len(optimizer_grouped_parameters)))  # 1

    if args.optimizer.lower()=="lamb":
        optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer.lower()=="adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        raise Exception("optimizer {0} not recognized! Can only be lamb or adamW".format(args.optimizer))
    
    if args.scheduler.lower()=="linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    elif args.scheduler.lower()=="cosine":
        scheduler = CosineAnnealingLR(optimizer, t_total, 1e-8)
    else:
        raise Exception("Scheduler {0} not recognized! Can only be linear or cosine".format(args.scheduler))

    # Check if saved optimizer or scheduler states exist
    # TODO: we find this consume huge amount of additional GPU memory with pytorch, thus disable for now
    # if os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")) and args.resume:
        # Load in optimizer and scheduler states
        # if is_first_worker():
        #     op_state = torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        #     print([len(x['params']) for x in op_state['param_groups']])
        #     real_op_state = optimizer.state_dict()
        #     print([len(x['params']) for x in real_op_state['param_groups']])
        # optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        # scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("   Train dataset size = %d", len(shuffled_fh))
    logger.info("   Num Epochs = %d", args.num_train_epochs)
    logger.info("   Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("   Total train batch size (w. parallel, distributed & accumulation) = %d", real_batch_size)
    logger.info("   Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("   Total optimization steps = %d", t_total)
    logger.info("   LR warmup steps = %d", args.warmup_steps)

    global_step = 0
    eval_cnt = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if (os.path.exists(args.model_name_or_path) and args.resume) or args.starting_step > 0:
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = args.starting_step

            if global_step <= 0:
                global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])

            epochs_trained = global_step // (args.expected_train_size // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (args.expected_train_size // args.gradient_accumulation_steps)

            logger.info("   Continuing training from checkpoint, will skip to saved global_step")
            logger.info("   Continuing training from epoch %d", epochs_trained)
            logger.info("   Continuing training from global step %d", global_step)
            logger.info("   Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except:
            logger.info("  Start training from a pretrained model") 

    tr_loss = 0.0

    tensorboard_scalars = {}
    model.zero_grad()

    eval_cfg = args.eval_configObj # this is also produced in the load_model_config() method
    eval_fn = wrapped_process_fn(tokenizer, args, eval_cfg)

    ideal_path = args.eval_ideal_path
    is_first_eval = (eval_cnt == 0)

    best_checkpoints = []
    set_seed(args)  # Added here for reproductibility

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    for m_epoch in train_iterator:
        # shuffle input after first epoch
        if m_epoch>0:
            shuffled_fh.change_seed(m_epoch)
        sds = SimplifiedStreamingDataset(shuffled_fh, train_fn, configObj.ix_func)
        train_dataloader = DataLoader(sds, batch_size=args.per_gpu_train_batch_size, num_workers=1)
        acc_accum = []
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), desc="Iteration", disable=args.local_rank not in [-1, 0]):
            if step % 100 == 0 and step > 0:
                logger.info('train_step: {}'.format(step))
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
          
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"query_ids": batch[0].long(), "query_attn_mask": batch[1].long(), 
                        "meta_ids": batch[3].long(), "meta_attn_mask": batch[4].long(),
                        "labels": batch[6].float()}
            
            # sync gradients only at gradient accumulation step
            if (step + 1) % args.gradient_accumulation_steps == 0:
                outputs = model(**inputs)
            else:
                with model.no_sync():
                    outputs = model(**inputs)
                    
            loss_combine = outputs[0]
            assert len(loss_combine) == 3
            loss = loss_combine["total_loss"]
            sim_combine = outputs[1]
            assert len(sim_combine) == 8
            acc = outputs[2]
            acc_accum.append(acc.item())

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    loss.backward()
                else:
                    with model.no_sync():
                        loss.backward()
            tr_loss += loss.item()

            if is_first_worker():
                print("unique labels: ", torch.unique(inputs["labels"]).int())
                print("Similarity combinations: ", sim_combine)

            for key, value in loss_combine.items():
                tensorboard_scalars[key] = tensorboard_scalars.setdefault(key, 0.0) + value.item()
            for key, value in sim_combine.items():
                tensorboard_scalars[key] = tensorboard_scalars.setdefault(key, 0.0) + value.mean().item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_during_training and global_step % (args.logging_steps_per_eval*args.logging_steps)==0:
                        if is_first_worker():
                            save_checkpoint(args, -1, model, tokenizer, logger=logger)

                        logger.info("Train acc: {}".format(sum(acc_accum)*1.0/len(acc_accum)))                        
                        
                        model.eval()
                        is_first_eval = (eval_cnt == 0)
                        args.global_step = global_step
                        init_time = time()
                        fidelity = eval_fidelity(args, model, eval_fn, eval_cfg.path, ideal_path, args.cache_dir, is_first_eval, args.eval_full, logger)
                        logger.info("Eval cost time: {}".format(time() - init_time))
                        eval_cnt+=1

                        model.train()

                        if is_first_worker():
                            if len(best_checkpoints)<3:
                                save_checkpoint(args, global_step, model, tokenizer, optimizer, scheduler, logger=logger)
                                best_checkpoints.append((global_step, fidelity))
                            else:
                                worst_checkpoint = sorted(best_checkpoints, key=lambda x: x[1])[0]
                                if fidelity>worst_checkpoint[1]:
                                    save_checkpoint(args, global_step, model, tokenizer, optimizer, scheduler, logger=logger)
                                    worst_cp_path = os.path.join(args.output_dir, "checkpoint-{}".format(str(worst_checkpoint[0])))
                                    shutil.rmtree(worst_cp_path)
                                    best_checkpoints.remove(worst_checkpoint)
                                    best_checkpoints.append((global_step, fidelity))
                                else:
                                    logger.info("Fidelity not in top 3!")
                                assert len(best_checkpoints)==3
                            tb_writer.add_scalar("fidelity", fidelity, global_step)

                            
                            logger.info("Fidelity: {0}".format(fidelity))
                        dist.barrier()

                    learning_rate_scalar = scheduler.get_lr()[0]

                    if is_first_worker():
                        tb_writer.add_scalar("learning_rate", learning_rate_scalar, global_step)
                        tb_writer.add_scalar("epoch", m_epoch, global_step)
                        for key, value in tensorboard_scalars.items():
                            tb_writer.add_scalar(key, value / args.logging_steps, global_step)
                        logger.info(json.dumps({**tensorboard_scalars, **{"step": global_step}}))
                    
                    tensorboard_scalars = {}
                    dist.barrier()

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        tb_writer.close()

    return global_step, tr_loss / global_step


def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_path",
        default="utmlb",
        type=str,
        help="path to the training data, or key for the train_path_dict in config",
    )

    parser.add_argument(
        "--eval_path",
        default="/webdata-nfs/kwtang/L1_data/eval_full.tsv",
        type=str,
        help="path to the eval data",
    )

    parser.add_argument(
        "--eval_ideal_path",
        default="/webdata-nfs/kwtang/L1_data/ideal_map_UN.tsv",
        type=str,
        help="path to the eval ideal mapping data",
    )

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_false", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )

    parser.add_argument(
        "--optimizer",
        default="lamb",
        type=str,
        help="Optimizer - lamb or adamW",
    )

    parser.add_argument(
        "--scheduler",
        default="linear",
        type=str,
        help="Scheduler - linear or cosine",
    )
    
    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--warmup_proportion", default=0.1, type=float, 
        help="Proportion of training to perform linear learning rate warmup for. E.g., 1/10 of training.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--logging_steps_per_eval", type=int, default=10, help="Eval every X logging steps.")
    
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle train data using args.seed",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from checkpoint step",
    )
    parser.add_argument(
        "--eval_full",
        action="store_true",
        help="Whether to eval full ranking fidelity",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--expected_train_size",
        default= 100000,
        type=int,
        help="Expected train dataset size",
    )

    parser.add_argument(
        "--starting_step",
        default= -1,
        type=int,
        help="starting step",
    )

    parser.add_argument(
        "--max_position",
        default= 2048,
        type=int,
        help="starting step",
    )

    parser.add_argument(
        "--num_heads",
        default= 12,
        type=int,
        help="number of attention heads of the layer",
    )

    parser.add_argument(
        "--seq_len",
        default= 1024,
        type=int,
        help="sequence length",
    )

    parser.add_argument(
        "--block",
        default= 16,
        type=int,
        help="block size",
    )

    parser.add_argument(
        "--num_random_blocks",
        default= 1,
        type=int,
        help="number of random blocks in each block row",
    )

    parser.add_argument(
        "--num_sliding_window_blocks",
        default= 3,
        type=int,
        help="number of blocks in sliding local attention window",
    )

    parser.add_argument(
        "--num_global_blocks",
        default= 1,
        type=int,
        help="how many consecutive blocks, starting from index 0, are considered as global attention",
    )

    parser.add_argument(
        "--different_layout_per_head",
        action="store_true",
        help="Whether each head should be assigned a different sparsity layout",
    )

    parser.add_argument(
        "--enable_sparse_transformer",
        action="store_true",
        help="Whether enable sparse transformer",
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
        
    args.cache_dir = os.path.join(args.output_dir, "cache")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.n_gpu_used = args.n_gpu
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.n_gpu = 1
        args.n_gpu_used = dist.get_world_size()
    args.device = device

    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    dist.barrier()

    log_path = os.path.join(args.output_dir, "spam_{0}.log".format(str(dist.get_rank())))
    logger = set_logger(log_path, args)


    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu_used,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    args.output_mode = "classification"
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.num_labels = num_labels

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config, tokenizer, model, configObj = load_model_config(args.model_type, args)
    
    if model.sim_weight:
        logger.info("Scale of sim-weight")
        # logger.info("Model w:" + str(model.w.data))
        # logger.info("Model b:" + str(model.b.data))
        logger.info("Model w:" + str(model.sim_weight))
        logger.info("Model b:" + str(model.sim_bias))

    if is_first_worker():
        # check that train tsv exists
        configObj.check()

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu_used)
        logger.info("per_gpu_train_batch_size: {} \t train_batch_size: {}".format(
            args.per_gpu_train_batch_size, train_batch_size))
        train_fn = wrapped_process_fn(tokenizer, args, configObj)

        with LineShuffler(configObj.path) as f:
            global_step, tr_loss = train(args, model, tokenizer, f, train_fn, configObj, logger)
        
        # Good practice: save your training arguments together with the trained model
        if is_first_worker():
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            logger.info("Saving model checkpoint to %s", args.output_dir)
            save_checkpoint(args, -1, model, tokenizer, logger=logger)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    dist.barrier()
    
    # Evaluation
    results = {}
    if args.do_eval:
        model.eval()
        eval_path = args.eval_path
        ideal_path = args.eval_ideal_path

        eval_cfg = args.eval_configObj
        eval_fn = wrapped_process_fn(tokenizer, args, eval_cfg)

        # don't generate cache here
        fidelity = eval_fidelity(args, model, eval_fn, eval_cfg.path, ideal_path, None, True, args.eval_full)

        if is_first_worker():
            logger.info("fidelity: {0}".format(str(fidelity)))
    dist.barrier()
    logger.info("Exiting...")

    return results


if __name__ == "__main__":
    main()
