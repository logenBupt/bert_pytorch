import os, errno
from L1.L1_models import *
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    XLMRobertaTokenizer,
    BertConfig,
    RobertaConfig,
    XLMRobertaConfig,
)
from L1.process_fn import L1_process_fn, dual_ix_func, GetTrainingDataProcessingFn, GetTripletTrainingDataProcessingFn, L1_roberta_process_fn, L1_process_fn_exp

# default_train_path = "/webdata-nfs/kwtang/L1_data/train_full.tsv"
# cb_train_path_2 = "/webdata-nfs/lexion/L1_train/train_cb.shuf.tsv"
# cb_train_path = "/webdata-nfs/kwtang/L1/train_cleanbody.shuf.tsv"

train_path_dict = {
                    "utmlb" : ("/webdata-nfs/kwtang/L1/train_cleanbody.shuf.tsv", # train
                            "/webdata-nfs/kwtang/L1/eval_full.tsv" # valid set (7M) + cleanbody
                            ), # since AUTCML + B is a superset of AUTCML, there is no need to support AUTCML only schema
                }

default_chunk_cfg = [[("query", "market"), (20, 5)], [("title", "anchor", "url", "click", "desc", "lang"),(20, 20, 20, 68, 50, 2)],]
UTMB_Short_chunk_cfg = [[("query",), (20,)], [("url", "title", "desc", "cleanbody"),(20, 20, 50, 100)],]
UTMB_chunk_cfg = [[("query", "market"), (20,5)], [("title", "url", "desc", "cleanbody"),(20, 20, 50, 300)],]

Body_chunk_cfg = [[("query",), (20,)], [("cleanbody",), (1023,)]]

train_schema_AUTCMLB = {"label": 0, "query" : 3, "title": 4, "anchor" : 5, "url" : 6, "click" : 7, "desc" : 8, "cleanbody" : 9, "market" : 10, "lang" : 11}
eval_schema_AUTCMLB = {"qid" : 0, "docid" : 1, "query" : 4, "title" : 5, "anchor": 6, "url" : 7, "click" : 8, "desc" : 9, "rating" : 10, "cleanbody" : 11, "market" : 12, "lang" : 13}

def wrapped_process_fn(tokenizer, args, configObj):
    def fn(line, i):
        return configObj.process_fn(line, i, tokenizer, args, configObj.map, configObj.chunk_cfg)
    return fn

class L1InputConfig:
    def __init__(self, name, model, path = None, process_fn = L1_process_fn, chunk_cfg=default_chunk_cfg, ix_func=None, tokenizer_class=BertTokenizer, config_class=BertConfig, use_mean=True, num_hidden_layers=3, col_map = train_schema_AUTCMLB):
        self.name = name
        self.path = path
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class
        self.process_fn = process_fn
        self.ix_func = ix_func
        self.chunk_cfg = chunk_cfg
        self.use_mean = use_mean
        self.map = {}
        self.fields = ["qid", "query", "title", "anchor", "url", "click", "desc", "label", "rating", "docid", "market", "lang", "cleanbody"]
        for k in self.fields:
            self.map[k] = -1
        for k, v in col_map.items():
            if k in self.map:
                self.map[k] = v
        essential_fields = ["query"]
        for k in essential_fields:
            if self.map[k]<0:
                raise Exception("{0} is essential but missing".format(k))

    def check(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)


configs = [
    L1InputConfig(name = "l1_original",
                model = L1_Original,
                chunk_cfg = Body_chunk_cfg,
                ),

    # L1InputConfig(name = "l1_original",
    #             model = L1_Original,

    #             ),

    L1InputConfig(name = "l1_original_100d",
                model = L1_Orig_100d
                ),
    
    L1InputConfig(name = "l1_100d_cls_shared",
                model = L1_100d_CLS_Shared
                ),
    
    
    L1InputConfig(name = "l1_utmb_short",
            model = L1_100d_CLS_Shared,
            chunk_cfg = UTMB_Short_chunk_cfg
            ),

    L1InputConfig(name = "l1_utmb_short_exp",
            model = L1_100d_CLS_Shared,
            chunk_cfg = UTMB_Short_chunk_cfg,
            process_fn = L1_process_fn_exp
            ),

    L1InputConfig(name = "ms_utmb_short_exp",
            tokenizer_class = RobertaTokenizer,
            config_class = RobertaConfig,
            model = RobertaDot_NLL_LN,
            chunk_cfg = UTMB_Short_chunk_cfg,
            process_fn = L1_process_fn_exp
            ),

    L1InputConfig(name = "ms_xlmr_utmb_short_exp",
            tokenizer_class = XLMRobertaTokenizer,
            config_class = XLMRobertaConfig,
            model = XLMRobertaDot_NLL_COS,
            chunk_cfg = UTMB_Short_chunk_cfg,
            process_fn = L1_process_fn_exp
            ),

    L1InputConfig(name = "ms_xlmr_utmb_exp",
            tokenizer_class = XLMRobertaTokenizer,
            config_class = XLMRobertaConfig,
            model = XLMRobertaDot_NLL_COS,
            chunk_cfg = UTMB_chunk_cfg,
            process_fn = L1_process_fn_exp
            ),

]

L1ConfigDict = {cfg.name:cfg for cfg in configs}

def load_model_config(model_type, args):
    configObj = L1ConfigDict[model_type]

    if args.train_path.lower() in train_path_dict:
        configObj.path = train_path_dict[args.train_path.lower()][0]
        args.eval_path = train_path_dict[args.train_path.lower()][1]
    else:
        configObj.path = args.train_path

    eval_config = L1InputConfig("eval", configObj.model_class, 
                                path = args.eval_path, 
                                process_fn =  L1_process_fn, # we found using original proc fn in eval works the best, something pending more in depth study
                                chunk_cfg = configObj.chunk_cfg, 
                                col_map = eval_schema_AUTCMLB)

    model_args = type('', (), {})()
    model_args.use_mean = configObj.use_mean
    model_args.max_position = args.max_position
    model_args.enable_sparse_transformer = args.enable_sparse_transformer
    model_args.num_heads = args.num_heads
    model_args.seq_len = args.seq_len
    model_args.use_MLP = False
    model_args.compressor_dim = 64
    model_args.nce_weight = 0.9

    # use default cache directory to avoid double downloading of the pretrained checkpoints
    config = configObj.config_class.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
        #cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        #cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.enable_sparse_transformer:
        tokenizer = update_tokenizer_model_max_length(tokenizer, args.max_position)

    # model = configObj.model_class.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    #     #cache_dir=args.cache_dir if args.cache_dir else None,
    #     model_argobj=model_args,
    # )

    model = configObj.model_class(config, model_args)

    args.configObj = configObj
    args.eval_configObj = eval_config

    return config, tokenizer, model, configObj

def update_tokenizer_model_max_length(tokenizer, max_position):
    """This function updates the position embedding length of a tokenizer to a new max position.
    Arguments:
        tokenizer: required: a transformer tokenizer
        max_position: required: an integer determining new position embedding size
    Return:
        tokenizer: updated tokenizer; in which model maximum length has been extended based on new size
    """

    tokenizer.model_max_length = max_position
    tokenizer.init_kwargs['model_max_length'] = max_position
    print(f'updated tokenizer model max imum length to {max_position}')

    return tokenizer
