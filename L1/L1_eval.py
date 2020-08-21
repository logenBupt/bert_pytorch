import os
import sys
sys.path.append("..")
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from iterable_dataset import StreamingDataLoader, CachedStreamingDataLoader, all_gather_cpu, all_gather
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
import faiss
import logging

from util import (
    is_first_worker,
    concat_key,
    )

# logger = logging.getLogger(__name__)

def fullrank_fidelity(index, q_embs, qids, docids, ideal_df, minBLA, score_map):
    fidelity_map = {0: 31, 1: 15, 2:7, 3:3, 4:0}
    D, I = index.search(q_embs, minBLA*5)
    I = docids[I]
    knn_pkl = {"D": D, "I": I}
    all_knn_list = all_gather(knn_pkl)
    D_merged = concat_key(all_knn_list, "D", axis=1)
    I_merged = concat_key(all_knn_list, "I", axis=1)
    idx = np.argsort(D_merged, axis=1)[:, ::-1]
    sorted_I = np.take_along_axis(I_merged, idx, axis=1)
    container = []
    for i, (qid, row) in enumerate(zip(qids, sorted_I)):
        score = 0.0
        top_dids = []
        for did in row:
            if did in top_dids:
                continue
            else:
                top_dids.append(did)
                try:
                    rating = score_map.loc[(qid, did)]
                    score += fidelity_map[rating]
                except KeyError:
                    pass
            if len(top_dids)>=minBLA:
                break
        if is_first_worker() and i<3:
            print(f"{qid}, {len(top_dids)}: {top_dids}")
        container.append([qid, score])
    scores = pd.DataFrame(container, columns=["qid", "score"]).set_index("qid")
    joined = pd.concat([scores, ideal_df], axis=1).fillna(0.0)
    joined["fidelity"] = joined["score"]/joined["ideal"]
    dist.barrier()
    fidelity = joined["fidelity"].mean()
    return fidelity    

def eval_fidelity(args, model, fn, eval_path, ideal_path, data_cache_dir, is_first_eval, eval_full=True, logger=None):
    if not logger:
        raise ValueError("Please provide a logger for eval")
    eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # with open(eval_path, encoding="utf-8-sig") as f:
    with open(eval_path, encoding="utf-8") as f:
        eval_dataloader = CachedStreamingDataLoader(f, fn, batch_size=eval_batch_size, cache_dir=data_cache_dir, prefix="L1eval")
        container = []
        q_emb_container = []
        body_emb_container = []
        qid_container = []
        docid_container = []
        
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(eval_dataloader(is_first_eval), desc="Evaluating", disable=args.local_rank not in [-1, 0])):
                if i % 100 == 0:
                    logger.info("eval step: {}".format(i))
                model.eval()
                batch = tuple(t.to(args.device).long() for t in batch)
                assert len(batch)>=10

                inputs = {"query_ids": batch[0], "query_attn_mask": batch[1], 
                        "meta_ids": batch[3], "meta_attn_mask": batch[4],
                        "labels": batch[6]}
                qids = batch[7].unsqueeze(-1).cpu().numpy()
                ratings = batch[8].unsqueeze(-1).cpu().numpy()
                docids = batch[9].unsqueeze(-1).cpu().numpy()
                output = model(**inputs)
                sims = output[1]["eval"].detach().unsqueeze(-1).cpu().numpy()
                assert sims.shape == ratings.shape
                #t = torch.cat((qids, sims, ratings), dim=1)
                t = np.concatenate((qids, sims, ratings, docids), axis=1)
                container.append(t)

                if eval_full:
                    q_embs = output[3].detach().cpu().numpy()
                    body_embs = output[4].detach().cpu().numpy()
                    q_emb_container.append(q_embs)
                    docid_container.append(docids.squeeze())
                    body_emb_container.append(body_embs)
                    qid_container.append(qids.squeeze())

    if eval_full:
        q_embs = np.concatenate(q_emb_container, axis=0)
        qids = np.concatenate(qid_container, axis=0)
        docids = np.concatenate(docid_container, axis=0)
        pkl = {"embs": q_embs, "ids": qids}
        all_pkl = all_gather_cpu(pkl)
        q_embs = concat_key(all_pkl, "embs", axis=0)
        qids = concat_key(all_pkl, "ids", axis=0)
        assert q_embs.shape[0] == qids.shape[0]
        logger.info("Eval Q shapes: q_embs.shape {} qids.shape {}".format(q_embs.shape, qids.shape))
        qid2embs = {}
        for qid, q_emb in zip(qids, q_embs):
            if qid not in qid2embs:
                qid2embs[qid] = q_emb
            else:
                diff = qid2embs[qid]-q_emb
                if not np.allclose(q_emb, qid2embs[qid], atol=1e-6):
                    logger.info("Eval Max diff: {} qid {}".format(diff.max(), qid))
        qids = []
        q_embs = []
        for qid, q_emb in qid2embs.items():
            qids.append(qid)
            q_embs.append(q_emb)
        qids = np.array(qids)
        q_embs = np.asarray(q_embs)
        assert len(q_embs.shape)==2
        
        body_embs = np.concatenate(body_emb_container, axis=0)
        logger.info("Eval embs shape: q_embs {}  body_embs{}".format(q_embs.shape, body_embs.shape))

    ideal_df = pd.read_csv(ideal_path, sep='\t', names=["qid", "ideal"]).set_index("qid")
    logger.info("ideal_df.head(5): {}".format(ideal_df.head(5).to_string()))
    logger.info("idea_df.shape: {}".format(ideal_df.shape))

    eval_data = np.concatenate(container, axis=0) # [batch * batch_steps, 4]
    df = pd.DataFrame(eval_data, columns=["qid", "sim", "rating", "docid"])
    logger.info("Eval df.head(5).to_string() {}".format(df.head(5).to_string()))
    df = df.astype({'qid': 'int64', 'rating': 'int64', "docid": 'int64'})
    dfs = all_gather_cpu(df)
    df = pd.concat(dfs, axis=0)
    logger.info("eval data after gather and pd.concat shape: {}".format(df.shape))
    df.index = np.arange(len(df))

    if is_first_worker():
        store = pd.HDFStore(os.path.join(args.output_dir, 'store.h5'))
        store['df'] = df
        store.close()
    
    dist.barrier()

    if eval_full:
        nu = df.groupby(["qid", "docid"])["rating"].nunique()
        assert np.all(nu.values==1)
        ratings_map = df.groupby(["qid", "docid"])["rating"].min()

        dim = body_embs.shape[1]
        faiss.omp_set_num_threads(8)
        # cpu_index = faiss.index_factory(dim, "IVF8192,Flat")
        # # cpu_index = faiss.index_factory(dim, "HNSW32")
        # # cpu_index.hnsw.efSearch = 64
        # cpu_index.metric_type = faiss.METRIC_INNER_PRODUCT
        # cpu_index.train(body_embs)
        # logger.info("Done training")
        # cpu_index.add(body_embs)
        # cpu_index.nprobe = 10
        # logger.info("Done creating HNSW index")
        # dist.barrier()
        # ann_fidelity = fullrank_fidelity(cpu_index, q_embs, qids, docids, ideal_df, 20, ratings_map)
        # logger.info(f"ANN fidelity: {ann_fidelity}")
        # dist.barrier()
        # del cpu_index

        flat_index = faiss.IndexFlatIP(dim)
        # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, flat_index)
        # gpu_index_flat.add(body_embs)
        flat_index.add(body_embs)
        logger.info("Done creating Flat index")
        dist.barrier()
        flat_fidelity = fullrank_fidelity(flat_index, q_embs, qids, docids, ideal_df, 20, ratings_map)
        logger.info(f"Flat fidelity: {flat_fidelity}")
        dist.barrier()
        del flat_index

    scores = compute_scores(df, "qid", "docid", "sim", "rating", 20)
    joined = pd.concat([scores, ideal_df], axis=1).fillna(0.0)
    logger.info("Eval Lengths: {} {} {} {} {}".format(scores.shape, ideal_df.shape, joined.shape, scores.dtypes, ideal_df.dtypes))
    joined["fidelity"] = joined["score"]/joined["ideal"]

    if is_first_worker():
        mkt_container = []
        with open(eval_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                cells = line.strip().split("\t")
                qid = int(cells[0])
                market = cells[12].split("-")[0].lower()
                mkt_container.append([qid, market])
        mkt_df = pd.DataFrame(mkt_container, columns=["qid", "market"]).groupby("qid")["market"].first()
        jm = pd.concat([joined, mkt_df], axis=1)
        fidelity_by_mkt = pd.concat([jm.groupby("market")["fidelity"].mean(), jm.groupby("market")["score"].count()], axis=1).sort_values(by="score", ascending=False)
        logger.info("Eval fidelity_by_mkt.to_string() {}".format(print(fidelity_by_mkt.to_string())))
    dist.barrier()
    
    fidelity = joined["fidelity"].mean()
    if is_first_worker():
        logger.info("Eval Fidelity: {}".format(fidelity))
    return fidelity

def compute_scores(df: pd.DataFrame, qid_col, docid_col, sim_col, rating_col, minBLA):
    # 31*Perfect+15*Excellent+7*Good+3*Fair
    fidelity_map = {0: 31, 1: 15, 2:7, 3:3, 4:0}
    d = df.groupby([qid_col, docid_col])[sim_col].max().reset_index()
    d = d.groupby(qid_col).apply(lambda x: x.sort_values(by=[sim_col], ascending=False)[:minBLA][docid_col]).reset_index()[[qid_col, docid_col]]
    r = df.groupby([qid_col, docid_col])[rating_col].min()
    j = d.join(r, on=[qid_col, docid_col], how="inner")
    j["score"] = j[rating_col].map(fidelity_map)
    scores = j.groupby(qid_col)["score"].sum()
    return scores