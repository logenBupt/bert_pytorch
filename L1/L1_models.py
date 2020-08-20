import torch
from torch import nn
from transformers import (
    BertPreTrainedModel,
    XLMRobertaModel,
    BertModel,
    XLMRobertaModel,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
)
import torch.nn.functional as F
from torch_blocksparse import SparsityConfig, BertSparseSelfAttention, FixedSparsityConfig, DenseSparsityConfig, BigBirdSparsityConfig

class BertDenseLN(nn.Module):
    def __init__(self, config, outdim, num_hidden_layers, use_xavier=True, type="bert", p_dropout=0.0):
        super().__init__()
        if type=="bert":
            self.bert = BertModel.from_pretrained("bert-base-multilingual-cased", num_hidden_layers=num_hidden_layers)
        elif type=="xlmr":
            config.num_hidden_layers = 3
            # self.bert = XLMRobertaModel.from_pretrained("xlm-roberta-base", num_hidden_layers=num_hidden_layers)
            self.bert = XLMRobertaModel(config)
        else:
            raise KeyError(f"{type} is not a valid type!")
        self.embeddingHead = nn.Linear(config.hidden_size, outdim)
        self.norm = nn.LayerNorm(outdim)
        self.activation = nn.Tanh()
        if use_xavier:
            nn.init.xavier_uniform_(self.embeddingHead.weight)
        else:
            self.embeddingHead.weight.data.normal_(mean=0.0, std=0.02)
        self.embeddingHead.bias.data.fill_(0.0)
        self.emb_dropout = nn.Dropout(p=p_dropout)

    def extend_position_embedding(self, max_position):
        """This function extends the position embedding weights of a model loaded from a checkpoint.
        It assumes the new max position is bigger than the original max length.
        Arguments:
            model: required: a transformer model
            max_position: required: an integer determining new position embedding size
        Return:
            model: updated model; in which position embedding weights have been extended based on new size
        """

        original_max_position = self.bert.embeddings.position_embeddings.weight.size(0)
        assert max_position > original_max_position
        extend_multiples = max(1, max_position // original_max_position)
        embedding = nn.Embedding(max_position, self.bert.config.hidden_size)
        embedding.weight = nn.Parameter(self.bert.embeddings.position_embeddings.weight.repeat(extend_multiples, 1))
        # self.bert.embeddings.position_embeddings.weight.data = self.bert.embeddings.position_embeddings.weight.repeat(
        #     extend_multiples, 1)
        self.bert.embeddings.position_embeddings = embedding
        self.bert.config.max_position_embeddings = max_position

        print(f'Extended position embeddings to {original_max_position * extend_multiples}')

    def replace_self_attention_layer_with_sparse_self_attention_layer(self, config, layers, sparsity_config=SparsityConfig(num_heads=4, seq_len=1024)):
        """This function replaces the self attention layers in attention layer with sparse self attention.
        For sparsityConfig, refer to the config class.
        Arguments:
            config: required: transformer model config
            layers: required: transformer model attention layers
            sparsity_config: optional: this parameter determins sparsity pattern configuration; it is based on SparsityConfig class
        Return:
            layers: updated attention layers; in which self attention layers have been repleaced with DeepSpeed Sparse Self Attention layer.
        """

        for layer in layers:
            deepspeed_sparse_self_attn = BertSparseSelfAttention(config, sparsity_config)
            deepspeed_sparse_self_attn.query = layer.attention.self.query
            deepspeed_sparse_self_attn.key = layer.attention.self.key
            deepspeed_sparse_self_attn.value = layer.attention.self.value
            
            layer.attention.self = deepspeed_sparse_self_attn


    def replace_model_self_attention_with_sparse_self_attention(
        self,
        max_position,
        # SparsityConfig parameters needs to be set accordingly
        sparsity_config=SparsityConfig(num_heads=4,
                                    seq_len=1024)):
        """This function replaces the self attention layers in model encoder with sparse self attention.
        It currently supports bert and roberta model and can be easily extended to any other models following similar steps here.
        For sparsityConfig, refer to the config class.
        Arguments:
            model: required: a transformer model
            max_position: required: an integer determining new position embedding size
            sparsity_config: optional: this parameter determins sparsity pattern configuration; it is based on SparsityConfig class
        Return:
            model: updated model; in which self attention layer has been repleaced with DeepSpeed Sparse Self Attention layer.
        """

        self.bert.config.max_position_embeddings = max_position
        self.replace_self_attention_layer_with_sparse_self_attention_layer(
            self.bert.config,
            self.bert.encoder.layer,
            sparsity_config)

    def update_to_sparse_transformer(self, max_position, sparsity_config=SparsityConfig(num_heads=4, seq_len=1024)):
        self.extend_position_embedding(max_position)
        self.replace_model_self_attention_with_sparse_self_attention(max_position, sparsity_config)


    def forward(self, input_ids, attention_mask, use_cls, compress, use_tanh, normalize=True):
        outputs1 = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        if use_cls:
            first_token = outputs1[0][:,0]
        else:
            # use pooled first token
            first_token = outputs1[1]
        first_token = self.emb_dropout(first_token)       
        if compress:
            first_token = self.embeddingHead(first_token)
        if use_tanh:
            first_token = self.activation(first_token)
        if normalize:
            first_token = F.normalize(first_token, p=2, dim=1)
        return first_token

class L1TrainModel(BertPreTrainedModel):
    def __init__(self, config, model_argobj):
        super().__init__(config)
        self.use_mean = False
        self.p_dropout = 0.1
        print("Using mean:", self.use_mean)
        self.bce_logit_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.num_hidden_layers = 3
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased", num_hidden_layers=self.num_hidden_layers)  
        self.hidden_size = config.hidden_size
        self.emb_dropout = nn.Dropout(p=self.p_dropout)
        if self.num_hidden_layers==3:
            self._load_layer_weights()
            print("Loaded weights from layer 4 and 8.")

    def _load_layer_weights(self):
        bert_full = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.bert.encoder.layer[1].load_state_dict(bert_full.encoder.layer[4].state_dict())
        self.bert.encoder.layer[2].load_state_dict(bert_full.encoder.layer[8].state_dict())

    def masked_mean(self, t, mask):
        s = torch.sum(t*mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s/d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:,0]

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        return self.masked_mean_or_first(outputs1, attention_mask)

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, query_ids, query_attn_mask, meta_ids, meta_attn_mask, labels):
        labels = labels.float()
        q_embs = self.query_emb(query_ids, query_attn_mask)
        body_embs = self.body_emb(meta_ids, meta_attn_mask)

        q_embs_norm = F.normalize(q_embs, p=2, dim=1)
        x_q_sim = torch.matmul(q_embs_norm, q_embs_norm.t())
        labels_repeated = labels.squeeze().unsqueeze(0).repeat(q_embs.shape[0],1)
        assert labels_repeated.shape==x_q_sim.shape
        x_q_mask = torch.where((x_q_sim>=0.9) & (labels_repeated==1.0), -1e9*torch.ones_like(x_q_sim), torch.zeros_like(x_q_sim))

        q_embs = self.emb_dropout(q_embs)
        body_embs = self.emb_dropout(body_embs)

        eval_sim = (q_embs*body_embs).sum(-1)
        eval_logits = eval_sim
        eval_loss = self.bce_logit_loss(eval_logits, labels)

        x_sim = torch.matmul(q_embs, body_embs.t())         
        x_sim_masked = x_sim + -1e12*torch.eye(x_sim.shape[0]).to(x_sim.device) + x_q_mask.to(x_sim.device)
        m_sample = 1
        nce_topk_idx = torch.topk(x_sim_masked, x_sim_masked.shape[-1])[1].detach()[:, :m_sample]
        neg_embs = body_embs[nce_topk_idx].reshape(-1, body_embs.shape[-1])
        repeated_q_embs = torch.repeat_interleave(q_embs, m_sample, dim=0)
        nce_sim = (repeated_q_embs*neg_embs).sum(-1)
        nce_logits = nce_sim
        nce_labels = torch.zeros_like(labels).to(nce_sim.device).float().repeat_interleave(m_sample)
        nce_weight = m_sample*1.0/(m_sample+1.0)
        
        assert len(nce_logits.shape)==1
        nce_loss = self.bce_logit_loss(nce_logits, nce_labels)
        loss = (nce_weight*nce_loss).mean() + ((1.0-nce_weight)*eval_loss).mean()

        preds = (eval_sim>0).int()
        acc = torch.eq(preds, labels.int()).float().mean()
        return loss, eval_logits, acc, q_embs, body_embs

class L1_Original(BertPreTrainedModel):
    def __init__(self, config, model_argobj):
        super().__init__(config)
        self.q_encoder = BertDenseLN(config, 768, num_hidden_layers=3, use_xavier=False, p_dropout=0.1)
        self.body_encoder = BertDenseLN(config, 768, num_hidden_layers=3, use_xavier=False, p_dropout=0.1)
        if model_argobj.enable_sparse_transformer:
            self.body_encoder.update_to_sparse_transformer(model_argobj.max_position, FixedSparsityConfig(model_argobj.num_heads, model_argobj.seq_len))
        self.q_encoder.bert.embeddings.word_embeddings = self.body_encoder.bert.embeddings.word_embeddings
        # self.w = torch.nn.Parameter(torch.ones(1)*10.0)
        # self.b = torch.nn.Parameter(torch.zeros(1))
        self.w = 10.0
        self.b = -5.0
        self.m_sample = 2
        self.bce_logit_loss = nn.BCEWithLogitsLoss(pos_weight=torch.full([1], self.m_sample, dtype=torch.float32), reduction='none')

    def query_emb(self, input_ids, attention_mask):
        print("query input_ids: ", input_ids.size())
        print("query attention_mask: ", attention_mask.size())
        q_embs_norm = self.q_encoder(input_ids, attention_mask, use_cls=True, compress=False, use_tanh=False, normalize=True)
        print("query embs norm: ", q_embs_norm.size())
        return q_embs_norm

    def body_emb(self, input_ids, attention_mask):
        print("body input_ids: ", input_ids.size())
        print("body attention_mask: ", attention_mask.size())
        print("body part config: ", self.body_encoder.bert.config)
        print("body part position_embeddings: ", self.body_encoder.bert.embeddings.position_embeddings)
        print("body part position_embeddings weight shape: ", self.body_encoder.bert.embeddings.position_embeddings.weight.data.size())
        body_embs_norm = self.body_encoder(input_ids, attention_mask, use_cls=True, compress=False, use_tanh=False, normalize=True)
        print("body embs norm: ", body_embs_norm.size())
        return body_embs_norm
    
    def forward(self, query_ids, query_attn_mask, meta_ids, meta_attn_mask, labels):
        labels = labels.float()
        body_embs_norm = self.body_emb(meta_ids, meta_attn_mask)
        q_embs_norm = self.query_emb(query_ids, query_attn_mask)

        x_q_sim = torch.matmul(q_embs_norm, q_embs_norm.t())
        x_q_mask = torch.where(x_q_sim>=0.9, -1e12*torch.ones_like(x_q_sim), torch.zeros_like(x_q_sim))

        eval_sim = (q_embs_norm*body_embs_norm).sum(-1)
        eval_logits = self.w*eval_sim+self.b
        eval_loss = self.bce_logit_loss(eval_logits, labels).mean()
        # eval_loss = self.bce_loss(eval_sim, labels).mean()
        loss = eval_loss

        if self.training:
            x_sim = torch.matmul(q_embs_norm, body_embs_norm.t())
            x_sim_masked = x_sim + x_q_mask + -1e12*torch.eye(x_sim.shape[0]).to(x_sim.device) # self-mask + mask queries that are too similar
            
            #nce_topk_idx = torch.argmax(x_sim_masked, dim=1).detach()
            nce_topk_idx = torch.topk(x_sim_masked, x_sim_masked.shape[-1])[1].detach()[:, :self.m_sample]
            neg_embs = body_embs_norm[nce_topk_idx].reshape(-1, body_embs_norm.shape[-1])
            assert len(neg_embs.shape)==2
            assert neg_embs.shape[0] == body_embs_norm.shape[0]*self.m_sample
            repeated_q_embs = torch.repeat_interleave(q_embs_norm, self.m_sample, dim=0)
            nce_sim = (repeated_q_embs*neg_embs).sum(-1)
            nce_labels = torch.zeros_like(labels).to(nce_sim.device).float().repeat_interleave(self.m_sample)
            nce_logits = self.w*nce_sim+self.b
            # nce_weight = self.m_sample*1.0/(self.m_sample+1.0)
            nce_weight = 0.5
            # assert len(nce_logits.shape)==1
            nce_loss = self.bce_logit_loss(nce_logits, nce_labels).mean()
            # nce_loss = self.bce_loss(nce_sim, nce_labels).mean()
            loss = nce_weight*nce_loss + (1.0-nce_weight)*eval_loss

        preds = (eval_logits>0).int()
        acc = torch.eq(preds, labels.int()).float().mean()  

        return loss, eval_logits, acc, q_embs_norm, body_embs_norm

class L1_Orig_CLS(L1_Original):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        q_embs_norm = F.normalize(outputs1[0][:,0], p=2, dim=1)
        return q_embs_norm

    def body_emb(self, input_ids, attention_mask):
        outputs1 = self.bert_body(input_ids=input_ids,
                            attention_mask=attention_mask)
        body_embs_norm = F.normalize(outputs1[0][:,0], p=2, dim=1)
        return body_embs_norm

class L1_100d_CLS_Shared(L1_Original):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)
        self.encoder = BertDenseLN(config, 100, num_hidden_layers=3, use_xavier=True)

    def query_emb(self, input_ids, attention_mask):
        q_embs_norm = self.encoder(input_ids, attention_mask, use_cls=True, compress=True, use_tanh=False, normalize=True)
        return q_embs_norm

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

class L1_Orig_100d(L1_Original):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)
        self.q_encoder = BertDenseLN(config, 100, num_hidden_layers=3)
        self.body_encoder = BertDenseLN(config, 100, num_hidden_layers=3)
        self.q_encoder.bert.embeddings.word_embeddings = self.body_encoder.bert.embeddings.word_embeddings            

    def query_emb(self, input_ids, attention_mask):
        q_embs_norm = self.q_encoder(input_ids, attention_mask, use_cls=False, compress=True, use_tanh=True, normalize=True)
        return q_embs_norm

    def body_emb(self, input_ids, attention_mask):
        body_embs_norm = self.body_encoder(input_ids, attention_mask, use_cls=False, compress=True, use_tanh=True, normalize=True)
        return body_embs_norm

class L1_Orig_100d_BCE(L1_Orig_100d):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)

    def forward(self, query_ids, query_attn_mask, meta_ids, meta_attn_mask, labels):
        labels = labels.float()
        q_embs_norm = self.query_emb(query_ids, query_attn_mask)
        body_embs_norm = self.body_emb(meta_ids, meta_attn_mask)

        eval_sim = (q_embs_norm*body_embs_norm).sum(-1)
        eval_logits = self.w*eval_sim+self.b
        eval_loss = self.bce_logit_loss(eval_logits, labels).mean()
        loss = eval_loss

        preds = (eval_logits>0).int()
        acc = torch.eq(preds, labels.int()).float().mean()  

        return loss, eval_sim, acc, q_embs_norm, body_embs_norm


class L1_TopK_NLL_Model(L1TrainModel):
    def __init__(self, config, model_argobj):
        super().__init__(config, model_argobj)
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased", num_hidden_layers=12)
        self.top_k = 1

    def forward(self, query_ids, query_attn_mask, meta_ids, meta_attn_mask, labels):
        labels = labels.float()
        q_embs = self.query_emb(query_ids, query_attn_mask)
        body_embs = self.body_emb(meta_ids, meta_attn_mask)
        eval_sim = (q_embs*body_embs).sum(-1)
        eval_logits = eval_sim
        x_sim = torch.matmul(q_embs, body_embs.t()) #[B, B]
        x_sim_masked = x_sim + -1e12*torch.eye(x_sim.shape[0]).to(x_sim.device)
        # self_logits = torch.diag(x_sim).unsqueeze(1)
        self_logits = eval_logits.unsqueeze(1)
        topk_logits = torch.topk(x_sim_masked, x_sim_masked.shape[-1])[0][:, :self.top_k]
        logit_matrix = torch.cat([self_logits, topk_logits], dim=1)
        assert self_logits.shape[0]==topk_logits.shape[0]
        pos_indices = torch.where(labels==1.0)[0]
        pos_logit_matrix = logit_matrix[pos_indices]
        assert pos_logit_matrix.shape[1]==self.top_k+1
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0*lsm[:,0].squeeze()
        # if self.training:
        #     self_logits = eval_logits[::2].unsqueeze(1)
        #     topk_logits = eval_logits[1::2].unsqueeze(1)
        #     logit_matrix = torch.cat([self_logits, topk_logits], dim=1)
        #     # pos_indices = torch.where(labels==1.0)[0]
        #     # pos_logit_matrix = logit_matrix[pos_indices]
        #     # assert pos_logit_matrix.shape[1]==self.top_k+1
        #     lsm = F.log_softmax(logit_matrix, dim=1)
        #     loss = -1.0*lsm[:,0].squeeze()
        # else:
        #     loss = torch.zeros_like(labels).float().to(labels.device)
        
        # preds = (eval_sim>0).int()
        # acc = torch.eq(preds, labels.int()).float().mean()
        acc = (torch.topk(pos_logit_matrix, 1)[1]==0).float().mean()
        return loss.mean(), eval_logits, acc, q_embs, body_embs 


# ============================================================================================
# MSMarco warmup model
# ============================================================================================
from transformers import (
    BertPreTrainedModel, 
    RobertaModel,
    RobertaForSequenceClassification
)

class RobertaDot_NLL_LN(RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        super().__init__(config)
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean

        print("Using mean:", self.use_mean)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.w = torch.nn.Parameter(torch.ones(1)*10.0)
        self.b = torch.nn.Parameter(torch.zeros(1))
        self.bce_loss = nn.BCELoss(reduction='none')
        self.bce_logit_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        # query1 = self.norm(self.embeddingHead(full_emb))
        query1 = F.normalize(self.embeddingHead(full_emb), p=2, dim=1)
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    # copied from L1 original's forward
    def forward(self, query_ids, query_attn_mask, meta_ids, meta_attn_mask, labels):
        labels = labels.float()
        q_embs_norm = self.query_emb(query_ids, query_attn_mask)
        body_embs_norm = self.body_emb(meta_ids, meta_attn_mask)

        x_q_sim = torch.matmul(q_embs_norm, q_embs_norm.t())
        x_q_mask = torch.where(x_q_sim>=0.9, -1e12*torch.ones_like(x_q_sim), torch.zeros_like(x_q_sim))

        eval_sim = (q_embs_norm*body_embs_norm).sum(-1)
        eval_logits = self.w*eval_sim+self.b
        eval_loss = self.bce_logit_loss(eval_logits, labels).mean()
        eval_sim = (eval_sim+1.0)/2.0
        # eval_loss = self.bce_loss(eval_sim, labels).mean()
        loss = eval_loss

        if self.training:
            x_sim = torch.matmul(q_embs_norm, body_embs_norm.t())
            x_sim_masked = x_sim + x_q_mask + -1e12*torch.eye(x_sim.shape[0]).to(x_sim.device) # self-mask + mask queries that are too similar
            
            m_sample = 1
            nce_topk_idx = torch.argmax(x_sim_masked, dim=1).detach()
            #nce_topk_idx = torch.topk(x_sim_masked, x_sim_masked.shape[-1])[1].detach()[:, :m_sample]
            neg_embs = body_embs_norm[nce_topk_idx].reshape(-1, body_embs_norm.shape[-1])
            assert len(neg_embs.shape)==2
            assert neg_embs.shape[0] == body_embs_norm.shape[0]
            repeated_q_embs = torch.repeat_interleave(q_embs_norm, m_sample, dim=0)
            nce_sim = (repeated_q_embs*neg_embs).sum(-1)
            nce_labels = torch.zeros_like(labels).to(nce_sim.device).float().repeat_interleave(m_sample)
            nce_logits = self.w*nce_sim+self.b
            nce_sim = (nce_sim+1.0)/2.0
            nce_weight = m_sample*1.0/(m_sample+1.0)
            # nce_weight = 0.8
            # assert len(nce_logits.shape)==1
            nce_loss = self.bce_logit_loss(nce_logits, nce_labels).mean()
            # nce_loss = self.bce_loss(nce_sim, nce_labels).mean()
            loss = nce_weight*nce_loss + (1.0-nce_weight)*eval_loss

        preds = (eval_logits>0).int()
        acc = torch.eq(preds, labels.int()).float().mean()  

        return loss, eval_sim, acc, q_embs_norm, body_embs_norm


class XLMRobertaDot_NLL_COS(XLMRobertaForSequenceClassification):

    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        super().__init__(config)
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean

        self.embeddingHead = nn.Linear(config.hidden_size, 100)

        # learned loss parameters
        self.w = torch.nn.Parameter(torch.ones(1)*10.0)
        self.b = torch.nn.Parameter(torch.zeros(1))

        self.bce_loss = nn.BCELoss(reduction='none')
        self.bce_logit_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]
        
    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.embeddingHead(full_emb)
        query1 = F.normalize(query1, p=2, dim=1)
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    
    def forward(self, query_ids, query_attn_mask, meta_ids, meta_attn_mask, labels):
        labels = labels.float()
        q_embs_norm = self.query_emb(query_ids, query_attn_mask)
        body_embs_norm = self.body_emb(meta_ids, meta_attn_mask)

        x_q_sim = torch.matmul(q_embs_norm, q_embs_norm.t())
        x_q_mask = torch.where(x_q_sim>=0.9, -1e12*torch.ones_like(x_q_sim), torch.zeros_like(x_q_sim))

        eval_sim = (q_embs_norm*body_embs_norm).sum(-1)
        eval_logits = self.w*eval_sim+self.b
        eval_loss = self.bce_logit_loss(eval_logits, labels).mean()
        eval_sim = (eval_sim+1.0)/2.0
        # eval_loss = self.bce_loss(eval_sim, labels).mean()
        loss = eval_loss

        if self.training:
            x_sim = torch.matmul(q_embs_norm, body_embs_norm.t())
            x_sim_masked = x_sim + x_q_mask + -1e12*torch.eye(x_sim.shape[0]).to(x_sim.device) # self-mask + mask queries that are too similar
            
            m_sample = 1
            nce_topk_idx = torch.argmax(x_sim_masked, dim=1).detach()
            #nce_topk_idx = torch.topk(x_sim_masked, x_sim_masked.shape[-1])[1].detach()[:, :m_sample]
            neg_embs = body_embs_norm[nce_topk_idx].reshape(-1, body_embs_norm.shape[-1])
            assert len(neg_embs.shape)==2
            assert neg_embs.shape[0] == body_embs_norm.shape[0]
            repeated_q_embs = torch.repeat_interleave(q_embs_norm, m_sample, dim=0)
            nce_sim = (repeated_q_embs*neg_embs).sum(-1)
            nce_labels = torch.zeros_like(labels).to(nce_sim.device).float().repeat_interleave(m_sample)
            nce_logits = self.w*nce_sim+self.b
            nce_sim = (nce_sim+1.0)/2.0
            nce_weight = m_sample*1.0/(m_sample+1.0)
            # nce_weight = 0.8
            # assert len(nce_logits.shape)==1
            nce_loss = self.bce_logit_loss(nce_logits, nce_labels).mean()
            # nce_loss = self.bce_loss(nce_sim, nce_labels).mean()
            loss = nce_weight*nce_loss + (1.0-nce_weight)*eval_loss

        preds = (eval_logits>0).int()
        acc = torch.eq(preds, labels.int()).float().mean()  

        return loss, eval_sim, acc, q_embs_norm, body_embs_norm
