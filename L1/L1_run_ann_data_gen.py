import sys
import codecs
sys.path.append("..")

from os import listdir
from os.path import isfile, join

import argparse
import glob
import json
import logging
import os, shutil
import random
import time
import pytrec_eval
import math

from L1.config import load_stuff

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch import nn

# removed unnecessary clutter - import everything from lee
from L1.L1_ann_utils import StreamingDataset, EmbeddingCache, get_checkpoint_no, get_latest_ann_data
from L1.preproc import GetProcessingFn
from iterable_dataset import all_gather, all_gather_cpu

from lamb import Lamb
import random 

import transformers
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
    is_first_worker,
    concat_key,
    )

# ANN - active learning ------------------------------------------------------
import faiss
import gzip
import pickle

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


from sklearn.metrics import roc_curve, auc
import pandas as pd


def get_latest_checkpoint(args):
    if not os.path.exists(args.training_dir):
        return args.init_model_dir
    subdirectories = list(next(os.walk(args.training_dir))[1])
    num_start_pos = len("checkpoint-")
    checkpoint_nums = [get_checkpoint_no(s) for s in subdirectories if s[:num_start_pos] == "checkpoint-"]
    if len(checkpoint_nums) > 0:
        return os.path.join(args.training_dir, "checkpoint-" + str(max(checkpoint_nums))) + "/"
    return args.init_model_dir


def load_positive_ids(args):

    logger.info("Loading query_2_pos_docid")
    training_query_positive_id = {}
    query_positive_id_path = os.path.join(args.data_dir, "train_qrels.tsv")
    with open(query_positive_id_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docid, rel] in tsvreader:
            assert rel == "1"
            topicid = int(topicid)
            docid = int(docid)
            training_query_positive_id[topicid] = docid

    logger.info("Loading dev query_2_pos_docid")
    dev_query_positive_id = {}

    return training_query_positive_id, dev_query_positive_id


def load_model(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.num_labels = num_labels
    args.model_name_or_path = checkpoint_path
    args.task_name = "MSMarco"
    args.do_lower_case = False
    config, tokenizer, model, configObj = load_stuff(args.model_type, args)
    model.to(args.device)
    logger.info("Inference parameters %s", args)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
    return config, tokenizer, model


def InferenceEmbeddingFromStreamDataLoader(args, model, train_dataloader, is_query_inference = True, prefix ="", end_batch=-1):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_batch_size = args.per_gpu_eval_batch_size

    # Inference!
    logger.info("***** Running ANN Embedding Inference *****")
    logger.info("  Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    if args.local_rank != -1:
        dist.barrier()
    model.eval()

    for i, batch in tqdm(enumerate(train_dataloader), desc="Inferencing", disable=args.local_rank not in [-1, 0], position=0, leave=True):
        if end_batch>0 and i>=end_batch:
            break
        idxs = batch[3].detach().numpy() #[#B]

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0].long(), "attention_mask": batch[1].long()}
            if is_query_inference:
                embs = model.module.query_emb(**inputs)
            else:
                embs = model.module.body_emb(**inputs)

        embs = embs.detach().cpu().numpy()

        # check for multi chunk output for long sequence 
        if len(embs.shape) == 3:
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:,chunk_no,:])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)


    embedding = np.concatenate(embedding, axis=0)
    embedding2id = np.concatenate(embedding2id, axis=0)
    return embedding, embedding2id


# streaming inference
def StreamInferenceDoc(args, model, fn, prefix, f, is_query_inference = True, end_batch=-1):
    inference_batch_size = args.per_gpu_eval_batch_size #* max(1, args.n_gpu)
    inference_dataset = StreamingDataset(f, fn)
    inference_dataloader = DataLoader(inference_dataset, batch_size=inference_batch_size)

    if args.local_rank != -1:
        dist.barrier() # directory created

    _embedding, _embedding2id = InferenceEmbeddingFromStreamDataLoader(args, model, inference_dataloader, is_query_inference = is_query_inference, prefix = prefix, end_batch=end_batch)
    np.savez_compressed(os.path.join(args.output_dir, "{1}part_{0}.npz".format(str(dist.get_rank()), prefix)), emb=_embedding, pid=_embedding2id)

    print(_embedding.shape, _embedding2id.shape)
    logger.info("merging embeddings")

    return _embedding, _embedding2id

def dump_embeddings(args, checkpoint_path):
    config, tokenizer, model = load_model(args, checkpoint_path)
    train_query_collection_path = os.path.join(args.data_dir, "train-query")
    train_query_cache = EmbeddingCache(train_query_collection_path)

    passage_collection_path = os.path.join(args.data_dir, "passages")
    passage_cache = EmbeddingCache(passage_collection_path)

    with passage_cache as emb:
        passage_embedding, passage_embedding2id = StreamInferenceDoc(args, model, GetProcessingFn(args, query=False), "passage_", emb, is_query_inference = False)
    logger.info("***** Done passage inference *****")

    with train_query_cache as emb:
        query_embedding, query_embedding2id = StreamInferenceDoc(args, model, GetProcessingFn(args, query=True), "query_", emb, is_query_inference = True)
    logger.info("***** Done query inference *****")

def get_string(emb, qid, tokenizer):
    id, pl, tokens = emb[qid]
    assert id==qid
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens)[:pl])

def generate_new_ann(args, output_num, checkpoint_path, training_query_positive_id, dev_query_positive_id, query_embcache, passage_embcache):
    while True:
        try:
            config, tokenizer, model = load_model(args, checkpoint_path)
            break
        except:
            time.sleep(60)
            print("retry loading model")

    passage_embcache.change_seed(output_num)
    query_embcache.change_seed(output_num)
    model.eval()
    logger.info("***** inference of passages *****")
    num_batches_per_gpu = len(passage_embcache)//(args.per_gpu_eval_batch_size*dist.get_world_size())

    args.corpus_divider = math.max(math.min(args.corpus_divider, 1.0), 0.0)
    # only run embedding inference for half of the passages to speed up process
    passage_embedding, passage_embedding2id = StreamInferenceDoc(args, model, GetProcessingFn(args, query=False), "passage_", passage_embcache, is_query_inference = False, end_batch=num_batches_per_gpu*args.corpus_divider)
    logger.info("***** Done passage inference *****")
    pid2ix = {v:k for k, v in enumerate(passage_embedding2id)}

    # build index partition on each process
    dim = passage_embedding.shape[1]
    logger.info('passage embedding shape: ' + str(passage_embedding.shape))
    top_k = args.topk_training 
    faiss.omp_set_num_threads(32//dist.get_world_size())
    print(faiss.omp_get_max_threads())

    if args.flat_index:
        cpu_index = faiss.IndexFlatIP(dim)
    else:
        cpu_index = faiss.index_factory(dim, "IVF8192,Flat")
        cpu_index.train(passage_embedding)

    cpu_index.add(passage_embedding)
    logger.info("***** Done training Index *****")
    cpu_index.nprobe = 50
    logger.info("***** Done building ANN Index *****")
    dist.barrier()
    
    if args.flat_index:
        flat_index = cpu_index
    else:
        flat_index = faiss.IndexFlatIP(dim)
        flat_index.add(passage_embedding)
        
    logger.info("**** Done building flat index *****")

    dev_ndcg, num_queries_dev = 0.0, 0
    train_data_output_path = os.path.join(args.output_dir, "ann_training_data_" + str(output_num))
    debug_output_path = os.path.join(args.output_dir, "ann_debug_"+ str(output_num))

    with open(train_data_output_path, 'w') as f, open(debug_output_path, "w", encoding="utf-8") as debug_g:
        chunk_factor = args.ann_chunk_factor
        if chunk_factor <= 0:
            chunk_factor = 1
        num_batches_per_gpu = len(query_embcache)//(args.per_gpu_eval_batch_size*dist.get_world_size())
        batches_per_chunk = num_batches_per_gpu // chunk_factor
        end_idx = batches_per_chunk
        print("End idx:", end_idx)

        inference_dataset = StreamingDataset(query_embcache, GetProcessingFn(args, query=True))
        inference_dataloader = DataLoader(inference_dataset, batch_size=args.per_gpu_eval_batch_size)
        out_train_list = []
        for m_batch, batch in tqdm(enumerate(inference_dataloader), desc="Inferencing", disable=args.local_rank not in [-1, 0], position=0):
            if m_batch>end_idx:
                break
            qids = batch[3].detach().numpy() #[#B]
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0].long(), "attention_mask": batch[1].long()}
                embs = model.module.query_emb(**inputs)
            embs = embs.detach().cpu().numpy()
            # take only queries with positive passage, then collect
            pos_idx = [i for i,qid in enumerate(qids) if qid in training_query_positive_id]
            query_embedding = embs[pos_idx]
            q_chunk_qid = qids[pos_idx]

            tmp_obj = {"emb": query_embedding, "id": q_chunk_qid}
            objs = all_gather(tmp_obj)
            query_embedding = concat_key(objs, "emb", axis=0)
            q_chunk_qid = concat_key(objs, "id", axis=0)
            if m_batch==0:
                print(query_embedding.shape, q_chunk_qid.shape)
            D, I = cpu_index.search(query_embedding, top_k)
            I = passage_embedding2id[I]

            if m_batch%100==0:
                logger.info(f"***** Done querying ANN Index chunk {m_batch}*****")
            knn_pkl = {"D": D, "I": I}
            all_knn_list = all_gather(knn_pkl)
            del knn_pkl

            if m_batch==0:
                # we only do flat_index search to debug
                Df, If = flat_index.search(query_embedding, top_k)
                If = passage_embedding2id[If]

                knn_pkl_flat = {"D": Df, "I": If}
                all_knn_list_flat = all_gather(knn_pkl_flat)
                del knn_pkl_flat

            if is_first_worker():
                D_merged = concat_key(all_knn_list, "D", axis=1)
                I_merged = concat_key(all_knn_list, "I", axis=1)

                # idx = np.argsort(D_merged, axis=1)[:, ::-1]
                if not args.ann_measure_topk_mrr:
                    shuffled_ix = np.random.permutation(I_merged.shape[1])[:args.negative_sample + 1]
                    sub_I = np.take(I_merged, shuffled_ix, axis=1) 
                else:
                    top_idx = np.argsort(D_merged, axis=1)[:args.negative_sample + 1]
                    sub_I = np.take_along_axis(I_merged, top_idx, axis=1)
                assert sub_I.shape[0] == len(q_chunk_qid)
                assert sub_I.shape[1] == args.negative_sample+1

                if m_batch==0:
                    D_merged_flat = concat_key(all_knn_list_flat, "D", axis=1)
                    I_merged_flat = concat_key(all_knn_list_flat, "I", axis=1)
                    shuffled_ix_flat = np.random.permutation(I_merged_flat.shape[1])[:args.negative_sample + 1]
                    sub_I_flat = np.take(I_merged_flat, shuffled_ix_flat, axis=1)
                else:
                    sub_I_flat = [0]*len(sub_I)

                for i, (qid, row, row_flat) in enumerate(zip(q_chunk_qid, sub_I, sub_I_flat)):
                    pos_pid = training_query_positive_id[qid]
                    neg_pids = [x for x in row if x!=pos_pid][:args.negative_sample]
                    if m_batch==0:
                        neg_pids_flat = [x for x in row_flat if x!=pos_pid][:args.negative_sample]
                    f.write("{}\t{}\t{}\n".format(qid, pos_pid, ','.join(str(nid) for nid in neg_pids)))
                    # debug_g.write("{}\t{}\t{}\n".format(qid, pos_pid, ','.join(str(nid) for nid in neg_pids)))
                    # console might crash if encoding isnt set correctly. run export PYTHONIOENCODING=UTF-8 before running the script if that happens.
                    if m_batch==0 and i<100:
                        q1 = get_string(query_embcache, qid, tokenizer)
                        # print(q1)
                        debug_g.write(q1+"\n")
                        p1 = get_string(passage_embcache, pos_pid, tokenizer)
                        debug_g.write(p1+"\n-------------\n")
                        # print(p1)
                        # print("-------------")
                        for nid in neg_pids:
                            ns = get_string(passage_embcache, nid, tokenizer)
                            debug_g.write(ns+"\n")
                            # print(ns)
                        # print("-------------")
                        debug_g.write("-------------\n")
                        for nid in neg_pids_flat:
                            ns = get_string(passage_embcache, nid, tokenizer)
                            debug_g.write(ns+"\n")
                            # print(ns)
                        debug_g.write("===============\n")
                        # print("==============")

    logger.info("*****Done Constructing ANN Triplet *****")
    ndcg_output_path = os.path.join(args.output_dir, "ann_ndcg_" + str(output_num))
    with open(ndcg_output_path, 'w') as f:
        json.dump({'ndcg': dev_ndcg, 'checkpoint': checkpoint_path}, f)

    return dev_ndcg, num_queries_dev

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--training_dir",
        default=None,
        type=str,
        required=True,
        help="Training dir, will look for latest checkpoint dir in here",
    )
    parser.add_argument(
        "--init_model_dir",
        default=None,
        type=str,
        #required=True,
        help="Initial model dir, will use this if no checkpoint is found in model_dir",
    )
    parser.add_argument(
        "--last_checkpoint_dir",
        default="",
        type=str,
        help="Last checkpoint used, this is for rerunning this script when some ann data is already generated",
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
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training data will be written",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=False,
        help="The directory where cached data will be written",
    )
    parser.add_argument(
        "--end_output_num",
        default=-1,
        type=int,
        help="Stop after this number of data versions has been generated, default run forever",
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

    parser.add_argument(
        "--max_doc_character",
        default= 10000, 
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=128,
        type=int,
        help="The starting output file number",
    )

    parser.add_argument(
        "--ann_chunk_factor",
        default= 1, # for 500k queryes, divided into 100k chunks for each epoch
        type=int,
        help="devide training queries into chunks",
    )

    parser.add_argument(
        "--corpus_divider",
        default= 1.0, # for 500k queryes, divided into 100k chunks for each epoch
        type=float,
        help="only take such portion of training corpus",
    )

    parser.add_argument(
        "--topk_training",
        default= 10,
        type=int,
        help="top k from which negative samples are collected",
    )

    parser.add_argument(
        "--negative_sample",
        default= 2,
        type=int,
        help="at each resample, how many negative samples per query do I use",
    )

    parser.add_argument(
        "--ann_measure_topk_mrr",
        default = False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--flat_index",
        default = False,
        action="store_true",
        help="use flat FAISS index",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

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

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    #set_seed(args)

    last_checkpoint = args.last_checkpoint_dir
    ann_no, ann_path, ndcg_json = get_latest_ann_data(args.output_dir)
    output_num = ann_no + 1

    logger.info("starting output number %d", output_num)

    if is_first_worker():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # dev_positive_id is empty
    training_positive_id, dev_positive_id = load_positive_ids(args)

    train_query_collection_path = os.path.join(args.data_dir, "train-query")
    train_query_cache = EmbeddingCache(train_query_collection_path)
    passage_collection_path = os.path.join(args.data_dir, "passages")
    passage_cache = EmbeddingCache(passage_collection_path)

    with train_query_cache as q_emb, passage_cache as p_emb:
        while args.end_output_num == -1 or output_num <= args.end_output_num:
            next_checkpoint = get_latest_checkpoint(args)
            if next_checkpoint == last_checkpoint:
                time.sleep(60)
            else:
                logger.info("start generate ann data number %d", output_num)
                logger.info("next checkpoint at " + next_checkpoint)
                generate_new_ann(args, output_num, next_checkpoint, training_positive_id, dev_positive_id, q_emb, p_emb)
                #dump_embeddings(args, next_checkpoint)
                logger.info("finished generating ann data number %d", output_num)
                output_num += 1
                last_checkpoint = next_checkpoint
            if args.local_rank != -1:
                dist.barrier()

if __name__ == "__main__":
    main()