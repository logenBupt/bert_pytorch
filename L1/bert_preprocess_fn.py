import torch

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

# def unicode_process(line, feature_col, default_feature=None):
#     feature = default_feature
#     try:
#         temp = line[feature_col]
#     except Exception:
#         temp = ""
#     if feature_col != -1:
#         feature = convert_to_unicode(temp)
#     return feature

def unicode_process(line, feature_col, default_feature=None):
    feature = default_feature
    if feature_col != -1:
        feature = convert_to_unicode(line[feature_col])
    return feature

def text_clean(text):
    text = text.replace('#n#', ' ').replace("<sep>", " ").replace('#tab#', ' ').replace('#r#', ' ').replace('\t', ' ')
    return " ".join(text.split())

max_length = {
    'query': 20,
    'url': 30,
    'title': 28,
    'meta_desc': 50,
    'body': 70,
}

def L1_bert_preprocess_fn(line, idx, tokenizer, args, col_map, max_length):
    """
    line: sample
    i: sample id
    col_map: feature and col_id map in dataset
        train_schema_AUTCMLB = {"label": 0, "query" : 3, "title": 4, "anchor" : 5, "url" : 6, "click" : 7, "desc" : 8, "cleanbody" : 9, "market" : 10, "lang" : 11}
        eval_schema_AUTCMLB = {"qid" : 0, "docid" : 1, "query" : 4, "title" : 5, "anchor": 6, "url" : 7, "click" : 8, "desc" : 9, "rating" : 10, "cleanbody" : 11, "market" : 12, "lang" : 13}
    """
    line = line.strip(' \n\r').split('\t')
    id_char_factor = 6

    src_col = col_map.get('query', -1)
    body_col = col_map.get('cleanbody', -1)
    

    query = unicode_process(line, src_col)

    body = text_clean(unicode_process(line, body_col)[:max_length['body']*id_char_factor])

    query_tokens_a = tokenizer.tokenize(query)

    query_tokens = []
    query_segment_ids = []
    query_tokens.append("[CLS]")
    query_segment_ids.append(0)
    if len(query_tokens_a) > max_length['query'] - 2:  # # [CLS]q0,q1...[SEP]
        query_tokens_a = query_tokens_a[0:(max_length['query'] - 2)]

    query_tokens.extend(query_tokens_a)
    query_segment_ids.extend([0] * len(query_tokens_a))
    query_tokens.append("[SEP]")
    query_segment_ids.append(0)

    metaStream_body = tokenizer.tokenize(body)
    if len(metaStream_body) > max_length['body'] - 2:
        metaStream_body = metaStream_body[0:(max_length['body'] - 2)]

    metaStream_tokens = []
    metaStream_segment_ids = []
    metaStream_tokens.append("[CLS]")
    metaStream_segment_ids.append(0)
    meta_append_segment_id = 0

    metaStream_tokens.extend(metaStream_body)
    metaStream_segment_ids.extend([meta_append_segment_id] * len(metaStream_body))
    metaStream_tokens.append("[SEP]")
    metaStream_segment_ids.append(meta_append_segment_id)
    meta_append_segment_id += 1

    
    # for i, item in enumerate(line):
    #     print(i, item)
    # print("query_tokens", query_tokens)
    # print("meta_tokens", metaStream_tokens)

    query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)
    meta_input_ids = tokenizer.convert_tokens_to_ids(metaStream_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    query_input_mask = [1] * len(query_input_ids)
    meta_input_mask = [1] * len(meta_input_ids)

    # Zero-pad up to the sequence length.
    if len(query_input_ids) < max_length['query']:
        pad_length = max_length['query'] - len(query_input_ids)
        query_input_ids.extend([0] * pad_length)
        query_input_mask.extend([0] * pad_length)
        query_segment_ids.extend([0] * pad_length)

    documents_max_len = max_length['body']

    if len(meta_input_ids) < documents_max_len:
        pad_length = documents_max_len - len(meta_input_ids)
        meta_input_ids.extend([0] * pad_length)
        meta_input_mask.extend([0] * pad_length)
        metaStream_segment_ids.extend([meta_append_segment_id] * pad_length)

    features = [
        torch.tensor(query_input_ids, dtype=torch.int), 
        torch.tensor(query_input_mask, dtype=torch.int), 
        torch.tensor(query_segment_ids, dtype=torch.uint8),
        torch.tensor(meta_input_ids, dtype=torch.int), 
        torch.tensor(meta_input_mask, dtype=torch.int), 
        torch.tensor(metaStream_segment_ids, dtype=torch.uint8),
    ]
    
    label = 0
    if col_map["label"] >= 0:   # positve sample is label==1 else neg sample
        label = int(line[col_map["label"]]) 
    features.append(label)
    if col_map["rating"] >= 0:  # what's rating?
        m = {"Perfect": 0, "Good": 2, "Fair": 3, "Excellent": 1, "Bad": 4}
        rating = m[line[col_map["rating"]]]
        docid = int(line[col_map["docid"]])
        qid = int(line[col_map["qid"]])
        features.extend([qid, rating, docid])
    features.append(idx)

    return [features]
