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
from multiprocessing import Process

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset, get_worker_info
from iterable_dataset import StreamingDataLoader

from lamb import Lamb
import random 

import transformers
import copy
import csv
from util import pad_input_ids, pad_ids, should_skip_rank, convert_to_string_id, pickle_save, pickle_load, barrier_list_merge, barrier_array_merge
from L1.L1_ann_utils import EmbeddingCache, numbered_byte_file_generator
import pickle
import gzip
from L1.process_fn import preprocess_fn, GetTrainingDataProcessingFn, GetTripletTrainingDataProcessingFn, GetProcessingFn
from L1.config import load_stuff, L1ConfigDict

from sklearn.metrics import roc_curve, auc
import pandas as pd

def tokenize_to_file(args, i, num_process, in_path, out_path, line_fn, chunk_cfg):
    tokenizer_class = args.configObj.tokenizer_class
    col_map = args.configObj.map
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=False,
        cache_dir=None,
    )
    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt', encoding='utf8') as in_f,\
            open('{}_split{}'.format(out_path, i), 'wb') as out_f:
        for idx, line in enumerate(in_f):
            if idx % num_process != i:
                continue
            out_f.write(line_fn(line, idx, tokenizer, args, chunk_cfg, **col_map))

def multi_file_process(args, num_process, in_path, out_path, line_fn, chunk_cfg):
    processes = []
    for i in range(num_process):
        p = Process(target=tokenize_to_file, args=(args, i, num_process, in_path, out_path, line_fn, chunk_cfg))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

def create_cache(in_path, out_path, chunk_cfg, args):
    multi_file_process(args, 32, in_path, out_path, preprocess_fn, chunk_cfg)
    assert len(chunk_cfg)==2
    max_len = sum(chunk_cfg[1])+1
    offset2id = []
    with open(out_path, 'wb') as f:
        for idx, record in enumerate(numbered_byte_file_generator(out_path, 32, 8 + 4 + max_len * 4)):
            p_id = int.from_bytes(record[:8], 'big')
            f.write(record[8:])
            offset2id.append(p_id)
            if idx < 3:
                print(str(idx) + " " + str(p_id))

    print("Total lines written: " + str(len(offset2id)))
    meta = {'type': 'int32', 'total_number': len(offset2id), 'embedding_size': max_len}
    with open(out_path + "_meta", 'w') as f:
        json.dump(meta, f)
    np.savez_compressed(out_path + '_idmap', idx=np.array(offset2id))
    embedding_cache = EmbeddingCache(out_path)
    print("First line")
    with embedding_cache as emb:
        for i, record in enumerate(emb):
            if i>0:
                break
            print(record)

def preprocess(args):
    args.num_labels= 2
    args.cache_dir = None
    args.task_name = "MSMarco"
    args.do_lower_case = False
    config, tokenizer, model, configObj = load_stuff(args.model_type, args)
    
    in_passage_path = os.path.join(args.data_dir, "train_full.tsv")
    train_qrels_path = os.path.join(args.data_dir, "train_qrels.tsv")

    out_passage_path = os.path.join(
        args.out_data_dir,
        "passages" ,
    )
    out_query_path = os.path.join(
        args.out_data_dir,
        "train-query" ,
    )
    out_qrels_path = os.path.join(
        args.out_data_dir,
        "train_qrels.tsv" ,
    )

    if os.path.exists(out_passage_path):
        print("preprocessed data already exist, exit preprocessing")
        return

    shutil.copyfile(train_qrels_path, out_qrels_path)

    print('start passage file split processing')

    assert len(configObj.chunk_cfg)>=2
    train_query_chunk_cfg = configObj.chunk_cfg[0]
    train_passage_chunk_cfg = configObj.chunk_cfg[1]
    # passage cache
    create_cache(in_passage_path, out_passage_path, train_passage_chunk_cfg, args)
    # query cache
    create_cache(in_passage_path, out_query_path, train_query_chunk_cfg, args)


def PassagePreprocessingFn(args, line, tokenizer):
    line_arr = line.split('\t')
    p_id = int(line_arr[0][1:]) # remove "D"

    url = line_arr[1].rstrip()
    title = line_arr[2].rstrip()
    p_text = line_arr[3].rstrip()

    full_text = url + "<sep>" + title + "<sep>" + p_text
    full_text = full_text[:args.max_doc_character] # keep only first 10000 characters, should be sufficient for any experiment that uses less than 500 - 1k tokens

    passage = tokenizer.encode(full_text, add_special_tokens=True, max_length= args.max_seq_length ,)
    passage_len = min(len(passage), args.max_seq_length)
    input_id_b = pad_input_ids(passage, args.max_seq_length)

    return p_id.to_bytes(8, 'big') + passage_len.to_bytes(4, 'big') + np.array(input_id_b, np.int32).tobytes()


def QueryPreprocessingFn(args, line, tokenizer):
    line_arr = line.split('\t')
    q_id = int(line_arr[0])

    passage = tokenizer.encode(line_arr[1].rstrip(), add_special_tokens=True, max_length=args.max_query_length)
    passage_len = min(len(passage), args.max_query_length)
    input_id_b = pad_input_ids(passage, args.max_query_length)

    return q_id.to_bytes(8, 'big') + passage_len.to_bytes(4, 'big') + np.array(input_id_b, np.int32).tobytes()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir",
    )
    parser.add_argument(
        "--out_data_dir",
        default=None,
        type=str,
        required=True,
        help="The output data dir",
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
    args = parser.parse_args()
    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)
    preprocess(args)

if __name__ == '__main__':
    main()
