import sys
sys.path += ['../']
import torch
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaTokenizer,

    # XLMR
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMRobertaConfig,
)
import torch.nn.functional as F
from data.process_fn import triple_process_fn, triple2dual_process_fn


class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

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
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class NLL(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


class NLL_MultiChunk(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        [batchS, full_length] = input_ids_a.size()
        chunk_factor = full_length // self.base_len

        # special handle of attention mask -----
        attention_mask_body = attention_mask_a.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(
            q_embs.unsqueeze(1), a_embs.transpose(
                1, 2))  # [batch, 1, chunk_factor]
        logits_a = (a12[:, 0, :] + inverted_bias).max(dim=-
                                                      1, keepdim=False).values  # [batch]
        # -------------------------------------

        # special handle of attention mask -----
        attention_mask_body = attention_mask_b.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(
            q_embs.unsqueeze(1), b_embs.transpose(
                1, 2))  # [batch, 1, chunk_factor]
        logits_b = (a12[:, 0, :] + inverted_bias).max(dim=-
                                                      1, keepdim=False).values  # [batch]
        # -------------------------------------

        logit_matrix = torch.cat(
            [logits_a.unsqueeze(1), logits_b.unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


class RobertaDot_CLF_ANN_NLL_MultiChunk(NLL_MultiChunk, RobertaDot_NLL_LN):
    def __init__(self, config):
        RobertaDot_NLL_LN.__init__(self, config)
        self.base_len = 512

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        chunk_factor = full_length // self.base_len

        input_seq = input_ids.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)
        attention_mask_seq = attention_mask.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq,
                                 attention_mask=attention_mask_seq)

        compressed_output_k = self.embeddingHead(
            outputs_k[0])  # [batch, len, dim]
        compressed_output_k = self.norm(compressed_output_k[:, 0, :])

        [batch_expand, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(
            batchS, chunk_factor, embeddingS)

        return complex_emb_k  # size [batchS, chunk_factor, embeddingS]
        
# -------------------------------------------------
# provide support for xlmr model as alternative for international coverage

class Cos_NLL(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)
        
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]

        # scaling factor
        lsm = F.log_softmax(self.w *logit_matrix + self.b, dim=1)
        # back to nll

        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


class Cos_BCE(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            labels = None,
            is_query=True):

        if input_ids_a is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_a is None:
            return self.body_emb(query_ids, attention_mask_q)
        
        labels = labels.float()


        q_embs_norm = self.query_emb(query_ids, attention_mask_q)
        body_embs_norm = self.body_emb(input_ids_a, attention_mask_a)
        
        bce_logit_loss = nn.BCEWithLogitsLoss(reduction='none')

        x_q_sim = torch.matmul(q_embs_norm, q_embs_norm.t())
        x_q_mask = torch.where(x_q_sim>=0.9, -1e12*torch.ones_like(x_q_sim), torch.zeros_like(x_q_sim))

        eval_sim = (q_embs_norm*body_embs_norm).sum(-1)
        eval_logits = self.w*eval_sim+self.b
        eval_loss = bce_logit_loss(eval_logits, labels).mean()
        # eval_loss = self.bce_loss(eval_sim, labels).mean()
        loss = eval_loss

        # NCE loss
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
        nce_loss = bce_logit_loss(nce_logits, nce_labels).mean()
        # nce_loss = self.bce_loss(nce_sim, nce_labels).mean()
        loss = nce_weight*nce_loss + (1.0-nce_weight)*eval_loss

        # preds = (eval_logits>0).int()
        # acc = torch.eq(preds, labels.int()).float().mean()  

        return (loss.mean(),)


class XLMRobertaDot_NLL_COS(Cos_NLL, XLMRobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        Cos_NLL.__init__(self, model_argobj)
        XLMRobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 100)

        # learned loss parameters
        self.w = torch.nn.Parameter(torch.ones(1)*10.0)
        self.b = torch.nn.Parameter(torch.zeros(1))

        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.embeddingHead(full_emb)
        query1 = F.normalize(query1, p=2, dim=1)
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

class XLMRobertaDot_BCE_COS(Cos_BCE, XLMRobertaDot_NLL_COS):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        Cos_BCE.__init__(self, model_argobj)
        XLMRobertaDot_NLL_COS.__init__(self, config)
        self.m_sample = 2


# # --------------------------------------------------
# ALL_MODELS = sum(
#     (
#         tuple(conf.pretrained_config_archive_map.keys())
#         for conf in (
#             RobertaConfig,
#             XLMRobertaConfig,
#         )
#     ),
#     (),
# )

MODEL_CLASSES = {
    "rdot_nll": (
        RobertaConfig,
        RobertaDot_NLL_LN,
        RobertaTokenizer),
    "rdot_nll_multi_chunk": (
        RobertaConfig,
        RobertaDot_CLF_ANN_NLL_MultiChunk,
        RobertaTokenizer),
        

    # xlmr
    "xlmrdot_nll": (
        XLMRobertaConfig,
        XLMRobertaDot_NLL_COS,
        XLMRobertaTokenizer),

}


default_process_fn = triple_process_fn


class MSMarcoConfig:
    def __init__(self, name, model, process_fn=default_process_fn, use_mean=True, tokenizer_class=RobertaTokenizer, config_class=RobertaConfig):
        self.name = name
        self.process_fn = process_fn
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class


configs = [
    MSMarcoConfig(name="rdot_nll",
                  model=RobertaDot_NLL_LN,
                  use_mean=False,
                  ),
]

MSMarcoConfigDict = {cfg.name: cfg for cfg in configs}
