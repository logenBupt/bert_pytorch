# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""
import sys
sys.path.append("..")

from os import listdir
from os.path import isfile, join

import argparse
import glob
import json
import logging
import os, shutil
import random
import pytrec_eval

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch import nn
import time

# removed unnecessary clutter - import everything from lee

from lamb import Lamb
import random 

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

    RobertaModel,
)

import copy
import csv
from torch import nn
import pickle

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from sklearn.metrics import roc_curve, auc
import pandas as pd

logger = logging.getLogger(__name__)

from util import (
    InputFeaturesPair, 
    getattr_recursive,
    barrier_array_merge,
    barrier_list_merge,
    pickle_save,
    pickle_load,
    pad_ids,
    convert_to_string_id,
    set_seed,
    )
from L1.L1_ann_utils import EmbeddingCache, get_checkpoint_no, get_latest_ann_data
from iterable_dataset import SimplifiedStreamingDataset
from L1.driver import save_checkpoint
from L1.config import L1InputConfig, L1ConfigDict, wrapped_process_fn, load_stuff
from L1.L1_eval import eval_fidelity
from L1.process_fn import L1_process_fn

def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def should_skip_rank(idx, args):
    if args.local_rank == -1:
        return False

    world_size = args.world_size
    return (idx % world_size) != args.rank


def train(args, model, tokenizer, query_cache, passage_cache):
    """ Train the model """
    #if args.local_rank in [-1, 0]:
    tb_writer = None
    if is_first_worker():
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    real_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    
    # layerwise optimization for lamb
    optimizer_grouped_parameters = []

    no_decay = ["bias", "w", "b", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]   

    if args.optimizer.lower()=="lamb":
        optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer.lower()=="adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        raise Exception("optimizer {0} not recognized! Can only be lamb or adamW".format(args.optimizer))

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and args.load_optimizer_scheduler:
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))

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
    #logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Max steps = %d", args.max_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    global_step = 0
    eval_cnt = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        if "-" in args.model_name_or_path:
            try:
                global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            except:
                global_step = 0
                
        else:
            global_step = 0
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from global step %d", global_step)

    tr_loss = 0.0
    model.zero_grad()
    model.train()
    set_seed(args)  # Added here for reproductibility

    last_ann_no = -1
    train_dataloader = None
    train_dataloader_iter = None
    dev_ndcg = 0
    step = 0

    eval_path = os.path.join(args.data_dir, "eval_full.tsv")
    eval_cfg = L1InputConfig("eval", model, None, eval_path, args.configObj.chunk_cfg, qid=0, docid=1, query=4, title=5, anchor=6, url=7, click=8, desc=9, rating=10, market=12, lang=13)

    def eval_fn(line, i):
        return L1_process_fn(line, i, tokenizer, args, eval_cfg.map, eval_cfg.chunk_cfg)

    model.eval()
    ideal_path = os.path.join(args.data_dir, "ideal_map_UN.tsv")
    is_first_eval = (eval_cnt == 0)
    
    args.global_step = global_step
    # fidelity = eval_fidelity(args, model, eval_fn, eval_cfg.path, ideal_path, args.data_cache_dir, is_first_eval)
    # eval_cnt+=1
    # print("Fidelity: {0}".format(fidelity))
    # if is_first_worker():
    #     tb_writer.add_scalar("fidelity", fidelity, global_step)

    best_checkpoints = []
    acc_accum = []
    scheduler = None
    while global_step < args.max_steps:
        if step % args.gradient_accumulation_steps == 0 and global_step % args.logging_steps == 0:
            # check if new ann training data is availabe
            ann_no, ann_path, ndcg_json = get_latest_ann_data(args.ann_dir)

            while ann_no == -1 or (ann_path is not None and ann_no != last_ann_no):
                try:
                    logger.info("Training on new add data at %s", ann_path)
                    with open(ann_path, 'r') as f:
                        ann_training_data = f.readlines()
                    dev_ndcg = ndcg_json['ndcg']
                    ann_checkpoint_path = ndcg_json['checkpoint']
                    ann_checkpoint_no = get_checkpoint_no(ann_checkpoint_path)

                    aligned_size = (len(ann_training_data) // args.world_size) * args.world_size
                    ann_training_data = ann_training_data[:aligned_size]

                    logger.info("Total ann queries: %d", len(ann_training_data))
                    if len(ann_training_data) == 0:
                        time.sleep(300)
                        continue

                    train_dataset = SimplifiedStreamingDataset(ann_training_data, args.configObj.process_fn(args, query_cache, passage_cache))
                    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=1)
                    train_dataloader_iter = iter(train_dataloader)

                    # re-warmup
                    if scheduler is None:
                        scheduler = get_linear_schedule_with_warmup(
                            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps= args.max_steps
                        )

                    if args.local_rank != -1:
                        dist.barrier()
            
                    if is_first_worker():
                        # add ndcg at checkpoint step used instead of current step
                        # tb_writer.add_scalar("dev_ndcg", dev_ndcg, ann_checkpoint_no)
                        if last_ann_no != -1:
                            tb_writer.add_scalar("epoch", last_ann_no, global_step-1)
                        tb_writer.add_scalar("epoch", ann_no, global_step)
                    last_ann_no = ann_no
                    break
                except:
                    if is_first_worker():
                        print("wait")
                    time.sleep(300)

                ann_no, ann_path, ndcg_json = get_latest_ann_data(args.ann_dir)

        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            logger.info("Finished iterating current dataset, begin reiterate")
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)
        
        batch = tuple(t.to(args.device).long() for t in batch)
        step += 1
        model.train()
        # if args.triplet:
        #     inputs = {"query_ids": batch[0].long(), "attention_mask_q": batch[1].long(), 
        #                 "input_ids_a": batch[3].long(), "attention_mask_a": batch[4].long(),
        #                 "input_ids_b": batch[6].long(), "attention_mask_b": batch[7].long()}
        # else:
        #     inputs = {"input_ids_a": batch[0].long(), "attention_mask_a": batch[1].long(), 
        #                 "input_ids_b": batch[3].long(), "attention_mask_b": batch[4].long(),
        #                 "labels": batch[6]}

        # sync gradients only at gradient accumulation step
        if step % args.gradient_accumulation_steps == 0:
            outputs = model(*batch)
        else:
            with model.no_sync():
                outputs = model(*batch)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
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
            if step % args.gradient_accumulation_steps == 0:
                loss.backward()
            else:
                with model.no_sync():
                    loss.backward()          

        tr_loss += loss.item()
        if step % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # print("w grad:", model.module.w.grad)
            # print("w:", model.module.w)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logs = {}
                if global_step % (args.logging_steps_per_eval*args.logging_steps)== 0:
                    print("Train acc:", sum(acc_accum)*1.0/len(acc_accum))
                    acc_accum = []
                    model.eval()
                    is_first_eval = (eval_cnt == 0)
                    args.global_step = global_step
                    fidelity = eval_fidelity(args, model, eval_fn, eval_cfg.path, ideal_path, args.data_cache_dir, is_first_eval)
                    eval_cnt+=1

                    if is_first_worker():
                        if len(best_checkpoints)<10:
                            save_checkpoint(args, global_step, model, tokenizer, optimizer, scheduler)
                            best_checkpoints.append((global_step, fidelity))
                        else:
                            worst_checkpoint = sorted(best_checkpoints, key=lambda x: x[1])[0]
                            if fidelity>worst_checkpoint[1]:
                                save_checkpoint(args, global_step, model, tokenizer, optimizer, scheduler)
                                worst_cp_path = os.path.join(args.output_dir, "checkpoint-{}".format(str(worst_checkpoint[0])))
                                shutil.rmtree(worst_cp_path)
                                best_checkpoints.remove(worst_checkpoint)
                                best_checkpoints.append((global_step, fidelity))
                            else:
                                print("Fidelity not in top 10!")
                            assert len(best_checkpoints)==10

                        save_checkpoint(args, -1, model, tokenizer)
                        print("Fidelity: {0}".format(fidelity))
                        logs["fidelity"] = fidelity
                    dist.barrier()
                loss_scalar = tr_loss / args.logging_steps
                learning_rate_scalar = scheduler.get_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                tr_loss = 0

                if is_first_worker():
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    logger.info(json.dumps({**logs, **{"step": global_step}}))


    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        tb_writer.close()

    return global_step


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain L1 eval files",
    )
    parser.add_argument(
        "--ann_dir",
        default=None,
        type=str,
        required=True,
        help="The ann training data dir. Should contain the output of ann data generation job",
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
        "--task_name",
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
        "--preproc_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory of preproc.py",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--data_cache_dir",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--triplet", default = False, action="store_true", help="Whether to run training.")

    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
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
        "--max_steps",
        default=1000000,
        type=int,
        help="If > 0: set total number of training steps to perform",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--logging_steps_per_eval", type=int, default=10, help="Eval every X logging steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    # ----------------- ANN HyperParam ------------------

    parser.add_argument(
        "--load_optimizer_scheduler",
        default = False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    # ----------------- End of Doc Ranking HyperParam ------------------
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

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
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    args.output_mode = "classification"
    #label_list = processor.get_labels()
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.num_labels = num_labels

    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    #args.model_type = args.model_type.lower()
    config, tokenizer, model, configObj = load_stuff(args.model_type, args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    query_collection_path = os.path.join(args.preproc_dir, "train-query")
    query_cache = EmbeddingCache(query_collection_path)
    passage_collection_path = os.path.join(args.preproc_dir, "passages")
    passage_cache = EmbeddingCache(passage_collection_path)

    with query_cache, passage_cache:
        global_step = train(args, model, tokenizer, query_cache, passage_cache)
        logger.info(" global_step = %s", global_step)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if is_first_worker():
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    
    if args.local_rank != -1:
        dist.barrier()


if __name__ == "__main__":
    main()