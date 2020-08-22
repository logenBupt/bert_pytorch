import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset, get_worker_info
from util import pad_ids

def preprocess_fn(line, i, tokenizer, args, chunk_cfg, **col_map):
    id_char_factor = 6
    mask_padding_with_zero = True
    chunk_pad_token_segment_id = 0
    pad_on_left = False
    pad_token_id = tokenizer.pad_token_id
    # ts = datetime.now()
    cells = line.strip().split("\t")
    if len(cells) >=8:
        chunk_ids = [tokenizer.cls_token_id]
        chunk_seg_ids = [0]
        max_chunk_len = 0
        for field, max_len in zip(*chunk_cfg):
            #print(field, col_map[field], len(cells))
            assert max_len>0
            assert col_map[field]>=0
            text = cells[col_map[field]].strip().lower()[:max_len*id_char_factor].replace("#n#"," ").replace("#tab#"," ")
            input_id_a = tokenizer.encode(text, add_special_tokens=False, max_length=max_len,truncation=True)
            if len(input_id_a)>=max_len:
                input_id_a[-1] = tokenizer.sep_token_id
            else:
                input_id_a.append(tokenizer.sep_token_id)
            token_type_ids = [0] * len(input_id_a)
            chunk_ids += list(input_id_a)
            chunk_seg_ids += list(token_type_ids)
            max_chunk_len += max_len
        assert len(chunk_ids)<=max_chunk_len+1, "{0} {1}".format(str(len(chunk_ids)), str(max_chunk_len+1))
        attention_mask = [1 if mask_padding_with_zero else 0] * len(chunk_ids)
        passage_len = len(chunk_ids)
        chunk_ids, chunk_attention_mask, chunk_seg_ids = pad_ids(chunk_ids, attention_mask, chunk_seg_ids, max_chunk_len+1, pad_token=pad_token_id, mask_padding_with_zero=mask_padding_with_zero, pad_token_segment_id=chunk_pad_token_segment_id, pad_on_left=pad_on_left)
        assert len(chunk_ids)==max_chunk_len+1, "{0} {1}".format(str(len(chunk_ids)), str(max_chunk_len+1))
    else:
        raise Exception("Line too short!")
    p_id = i
    return p_id.to_bytes(8, 'big') + passage_len.to_bytes(4, 'big') + np.array(chunk_ids, np.int32).tobytes()

def text_clean(text):
    text = text.replace('#n#', ' ').replace("<sep>", " ").replace('#tab#', ' ').replace('#r#', ' ').replace('\t', ' ')
    return " ".join(text.split())

def L1_process_fn(line, i, tokenizer, args, col_map, cfg):
    features = []
    id_char_factor = 6
    mask_padding_with_zero = True
    chunk_pad_token_segment_id = 0
    pad_on_left = False
    pad_token_id = tokenizer.pad_token_id
    # ts = datetime.now()
    cells = line.strip().split("\t")
    if len(cells) >=8:
        for chunk in cfg:
            chunk_ids = [tokenizer.cls_token_id]
            chunk_seg_ids = [0]
            max_chunk_len = 0
            for field, max_len in zip(*chunk):
                #print(field, col_map[field], len(cells))
                assert max_len>0
                text = text_clean(cells[col_map[field]].strip().lower()[:max_len*id_char_factor])
                input_id_a = tokenizer.encode(text, add_special_tokens=False, max_length=max_len,truncation=True)
                if len(input_id_a)>=max_len:
                    input_id_a[-1] = tokenizer.sep_token_id
                else:
                    input_id_a.append(tokenizer.sep_token_id)
                token_type_ids = [0] * len(input_id_a)
                chunk_ids += list(input_id_a)
                chunk_seg_ids += token_type_ids
                max_chunk_len += max_len
            assert len(chunk_ids)<=max_chunk_len+1, "{0} {1}".format(str(len(chunk_ids)), str(max_chunk_len+1))
            
            attention_mask = [1 if mask_padding_with_zero else 0] * len(chunk_ids)
            chunk_ids, chunk_attention_mask, chunk_seg_ids = pad_ids(chunk_ids, attention_mask, chunk_seg_ids, max_chunk_len+1, pad_token=pad_token_id, mask_padding_with_zero=mask_padding_with_zero, pad_token_segment_id=chunk_pad_token_segment_id, pad_on_left=pad_on_left)
            assert len(chunk_ids)==max_chunk_len+1, "{0} {1}".format(str(len(chunk_ids)), str(max_chunk_len+1))
            features += [torch.tensor(chunk_ids, dtype=torch.int), torch.tensor(chunk_attention_mask, dtype=torch.bool), torch.tensor(chunk_seg_ids, dtype=torch.uint8)]

        label = 0
        if col_map["label"]>=0:
            label = int(cells[col_map["label"]])
            # v = int(cells[col_map["label"]]) 
            # if v==1:
            #     label = 1
        features.append(label)
        if col_map["rating"]>=0:
            m = {"Perfect": 0, "Good": 2, "Fair": 3, "Excellent": 1, "Bad": 4}
            rating = m[cells[col_map["rating"]]]
            docid = int(cells[col_map["docid"]])
            qid = int(cells[col_map["qid"]])
            features += [qid, rating, docid]
        features.append(i)
    else:
        raise Exception("Line too short!")
        
    return [features]

# roberta func
def L1_roberta_process_fn(line, i, tokenizer, args, col_map, cfg):
    features = []
    id_char_factor = 6
    mask_padding_with_zero = True
    chunk_pad_token_segment_id = 0
    pad_on_left = False
    pad_token_id = tokenizer.pad_token_id
    # ts = datetime.now()
    cells = line.strip().split("\t")
    if len(cells) >=8:
        for chunk in cfg:
            all_text = None
            max_chunk_len = 0
            for field, max_len in zip(*chunk):
                
                assert max_len>0
                text = text_clean(cells[col_map[field]].strip().lower()[:max_len*id_char_factor])
                
                # print('--------------------------------------------------------------')
                # print(field)
                # print(cells[col_map[field]].strip().lower()[:max_len*id_char_factor])
                # print(text)

                # print('------------Finish Print--------------------------------------------------')

                if all_text is None:
                    all_text = text
                else:
                    all_text = all_text + '<sep>' + text

                max_chunk_len += max_len
            
            chunk_ids = tokenizer.encode(all_text, add_special_tokens=True, max_length=max_chunk_len,truncation=True)
            chunk_seg_ids = [0] * len(chunk_ids)

            # print("chunk len", len(chunk_ids))
            # print("max_chunk_len", max_chunk_len)
            
            attention_mask = [1 if mask_padding_with_zero else 0] * len(chunk_ids)
            chunk_ids, chunk_attention_mask, chunk_seg_ids = pad_ids(chunk_ids, attention_mask, chunk_seg_ids, max_chunk_len, pad_token=pad_token_id, mask_padding_with_zero=mask_padding_with_zero, pad_token_segment_id=chunk_pad_token_segment_id, pad_on_left=pad_on_left)
            # assert len(chunk_ids)==max_chunk_len+1, "{0} {1}".format(str(len(chunk_ids)), str(max_chunk_len+1))
            features += [torch.tensor(chunk_ids, dtype=torch.int), torch.tensor(chunk_attention_mask, dtype=torch.bool), torch.tensor(chunk_seg_ids, dtype=torch.uint8)]

        label = 0
        if col_map["label"]>=0:
            v = int(cells[col_map["label"]]) 
            if v==1:
                label = 1
        features.append(label)
        if col_map["rating"]>=0:
            m = {"Perfect": 0, "Good": 2, "Fair": 3, "Excellent": 1, "Bad": 4}
            rating = m[cells[col_map["rating"]]]
            docid = int(cells[col_map["docid"]])
            qid = int(cells[col_map["qid"]])
            features += [qid, rating, docid]
        features.append(i)
    else:
        raise Exception("Line too short!")
        
    return [features]

# ============================================
# experimental processing fn
# ============================================

def valid_sentence(sent):
    if len(sent) < 4:
        return False
    
    if (
        # '>' in sent
        '/' in sent
        or '<' in sent
        or '{' in sent
        or '}' in sent
        # or '-' in sent
        or '+' in sent
        or '=' in sent
        or '*' in sent
        or '@' in sent
        # or ':' in sent
        ) :
        return False

    num_count = sum(c.isdigit() for c in sent)
    num_space = sent.count(' ')
    num_co = sent.count(',')
    num_da = sent.count('-')
    num_q = sent.count('?')
    num_a = sent.count('>')
    
    if (num_count + num_space + num_co + num_da + num_q + num_a)> len(sent) / 3:
        return False

    if ("log" in sent
        or "comment" in sent
        or 'http' in sent
        or '.com' in sent
        or 'contact' in sent
        or 'loading' in sent ):

        return False

    return True


def text_clean_exp(text):
    text = text.replace('#tab#', ' ').replace('#r#', ' ').replace('#n#', '\t').replace('|', '\t').replace('-', ' - ').replace('\'', ' \' ').replace("<sep>", " ")
    text = " ".join(text.split()) # remove duplicated spaces

    sentences = text.split('\t')
    valid_sentences = []

    for sent in sentences:
        if valid_sentence(sent):
            valid_sentences.append(sent)

    sent_text = ' . '.join(valid_sentences)
    return sent_text

def L1_process_fn_exp(line, i, tokenizer, args, col_map, cfg):
    features = []
    id_char_factor = 6
    mask_padding_with_zero = True
    chunk_pad_token_segment_id = 0
    pad_on_left = False
    pad_token_id = tokenizer.pad_token_id
    # ts = datetime.now()
    cells = line.strip().split("\t")
    if len(cells) >=8:
        for chunk in cfg:
            chunk_ids = [tokenizer.cls_token_id]
            chunk_seg_ids = [0]
            max_chunk_len = 0
            for field, max_len in zip(*chunk):
                #print(field, col_map[field], len(cells))
                assert max_len>0
                text = cells[col_map[field]].strip().lower()[:max_len*id_char_factor]

                if field == 'cleanbody':
                    text = text_clean_exp(text)

                input_id_a = tokenizer.encode(text, add_special_tokens=False, max_length=max_len,truncation=True)
                if len(input_id_a)>=max_len:
                    input_id_a[-1] = tokenizer.sep_token_id
                else:
                    input_id_a.append(tokenizer.sep_token_id)
                token_type_ids = [0] * len(input_id_a)
                chunk_ids += list(input_id_a)
                chunk_seg_ids += token_type_ids
                max_chunk_len += max_len
            assert len(chunk_ids)<=max_chunk_len+1, "{0} {1}".format(str(len(chunk_ids)), str(max_chunk_len+1))
            
            attention_mask = [1 if mask_padding_with_zero else 0] * len(chunk_ids)
            chunk_ids, chunk_attention_mask, chunk_seg_ids = pad_ids(chunk_ids, attention_mask, chunk_seg_ids, max_chunk_len+1, pad_token=pad_token_id, mask_padding_with_zero=mask_padding_with_zero, pad_token_segment_id=chunk_pad_token_segment_id, pad_on_left=pad_on_left)
            assert len(chunk_ids)==max_chunk_len+1, "{0} {1}".format(str(len(chunk_ids)), str(max_chunk_len+1))
            features += [torch.tensor(chunk_ids, dtype=torch.int), torch.tensor(chunk_attention_mask, dtype=torch.bool), torch.tensor(chunk_seg_ids, dtype=torch.uint8)]

        label = 0
        if col_map["label"]>=0:
            v = int(cells[col_map["label"]]) 
            if v==1:
                label = 1
        features.append(label)
        if col_map["rating"]>=0:
            m = {"Perfect": 0, "Good": 2, "Fair": 3, "Excellent": 1, "Bad": 4}
            rating = m[cells[col_map["rating"]]]
            docid = int(cells[col_map["docid"]])
            qid = int(cells[col_map["qid"]])
            features += [qid, rating, docid]
        features.append(i)
    else:
        raise Exception("Line too short!")
        
    return [features]

    
def dual_ix_func(i, local_rank, world_size, record):
    return (i//2) % world_size == local_rank

def GetTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(args, query=True)(query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(args, query=False)(passage_cache[pos_pid], pos_pid)[0]

        pos_label = torch.tensor(1, dtype=torch.long)
        neg_label = torch.tensor(0, dtype=torch.long)
        yield (query_data[0], query_data[1], pos_data[0], pos_data[1], 1)

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(args, query=False)(passage_cache[neg_pid], neg_pid)[0]
            # yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2], pos_label)
            # yield (query_data[0], query_data[1], query_data[2], neg_data[0], neg_data[1], neg_data[2], neg_label)
            yield (query_data[0], query_data[1], neg_data[0], neg_data[1], 0)

    return fn


def GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(args, query=True)(query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(args, query=False)(passage_cache[pos_pid], pos_pid)[0]

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(args, query=False)(passage_cache[neg_pid], neg_pid)[0]
            # yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2],
            #         neg_data[0], neg_data[1], neg_data[2])
            yield (query_data[0], query_data[1], pos_data[0], pos_data[1],
                    neg_data[0], neg_data[1])

    return fn

def GetProcessingFn(args, query=False):
    def fn(vals, i):
        pid, passage_len, passage = vals
        # we need to change this
        max_len = len(passage)
        # max_len = args.max_query_length if query else args.max_seq_length
        
        pad_len = max(0, max_len - passage_len)
        token_type_ids = ([0] if query else [1]) * passage_len + [0] * pad_len
        attention_mask = [1] * passage_len + [0] * pad_len

        passage_collection = [(pid, passage, attention_mask, token_type_ids)]
        
        query2id_tensor = torch.tensor([f[0] for f in passage_collection], dtype=torch.long)
        all_input_ids_a = torch.tensor([f[1] for f in passage_collection], dtype=torch.int)
        all_attention_mask_a = torch.tensor([f[2] for f in passage_collection], dtype=torch.bool)
        all_token_type_ids_a = torch.tensor([f[3] for f in passage_collection], dtype=torch.uint8)

        dataset = TensorDataset(all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, query2id_tensor)

        return [ts for ts in dataset]
    
    return fn