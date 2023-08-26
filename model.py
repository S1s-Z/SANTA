import numpy as np
import math
import torch
from torch import nn
from transformers import BertModel,BertConfig,RobertaModel,RobertaConfig,AutoModel,AutoConfig
import torch.nn.functional as F
import random
from misc import flat_list
from misc import sim_matrix, contrastive_loss
from misc import iterative_support, conflict_judge
from utils import UnitAlphabet, LabelAlphabet
from sklearn.metrics.pairwise import cosine_similarity
import time
import faiss

class SupConLossPLMS(torch.nn.Module):
    def __init__(self, device=torch.cuda.current_device() , temperature=0.05):
        super(SupConLossPLMS, self).__init__()
        self.tem = temperature
        self.device = device

    def forward(self, batch_emb, labels, label_num, label_vocab, have_O=False):
        entity_index_set = np.argwhere(labels.cpu() != label_vocab.index('O')).reshape(-1)
        if entity_index_set.tolist() == [] or len(entity_index_set.tolist()) == 1:
            return torch.tensor(1).cuda().detach()
        else:
            pass
        labels = labels[entity_index_set]
        labels = labels.view(-1, 1)
        batch_emb = batch_emb[entity_index_set]

        index_mask = torch.torch.arange(label_num, labels.size()[0] + label_num).view(-1, 1).cuda()
        labels_mask = torch.where(labels > 0, labels, index_mask).cuda()
        batch_size = batch_emb.shape[0]
        mask = torch.eq(labels_mask, labels_mask.T).float().cuda()
        norm_emb = F.normalize(batch_emb, dim=1, p=2)
        dot_contrast = torch.div(torch.matmul(norm_emb, norm_emb.T), self.tem)
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        logits = dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_sum
        return mean_log_prob_pos.mean()


class pNorm(nn.Module):
    def __init__(self, p=0.3):
        super(pNorm, self).__init__()
        self.p = p

    def forward(self, pred, p=None):
        if p:
            self.p = p
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.5)

        norm = torch.sum(pred ** self.p, dim=1)
        return norm.mean()

def calculate_pNorm_loss(criterion, out, y, norm, lamb=0.1, tau=0.7, p=0.5):

    loss = criterion(out / tau, y) + lamb * norm(out / tau, p)

    return loss

class GCELoss(nn.Module):
    def __init__(self, num_classes, q=0.3):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

class FocalLoss(nn.Module):

    def __init__(self, num_classes, weight=None, reduction='mean', gamma=2, alpha=0.5, eps=1e-7, beta=0.2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.num_classes = num_classes
        self.beta = beta

    def forward(self, input, target, softlabel_matrix):
        onehot_target = torch.nn.functional.one_hot(target, num_classes = self.num_classes)
        smooth_target = torch.zeros_like(onehot_target)
        for index_smooth in range(0,len(target)):
            smooth_target[index_smooth] = softlabel_matrix[target[index_smooth]]

        logp = input.log_softmax(dim=-1)
        logp = logp.gather(dim=1, index=target.unsqueeze(-1)).squeeze(-1).mean()
        p = logp.exp()
        loss_soft = - (self.alpha * (1 - p)** self.gamma * (logp * smooth_target).sum(dim=-1) )
        loss_hard = - (self.alpha * (1 - p)** self.gamma * (logp * onehot_target).sum(dim=-1) )
        loss = (1-self.beta) * loss_hard + self.beta * loss_soft
        return loss.mean()

def relative_position_encoding(depth, max_length=512, max_relative_position=127):
    vocab_size = max_relative_position * 2 + 1
    range_vec = torch.arange(max_length)
    range_mat = range_vec.repeat(max_length).view(max_length, max_length)
    distance_mat = range_mat - torch.t(range_mat)
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position

    embeddings_table = torch.zeros(vocab_size, depth)
    position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth))
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    embeddings_table = embeddings_table.unsqueeze(0).transpose(0, 1).squeeze(1)

    flat_relative_positions_matrix = final_mat.view(-1)
    one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix,
                                                                    num_classes=vocab_size).float()
    positions_encoding = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
    my_shape = list(final_mat.size())
    my_shape.append(depth)
    positions_encoding = positions_encoding.view(my_shape)
    return positions_encoding

class PhraseClassifier(nn.Module):

    def __init__(self,
                 lexical_vocab: UnitAlphabet,
                 label_vocab: LabelAlphabet,
                 hidden_dim: int,
                 dropout_rate: float,
                 neg_rate: float,
                 clloss_percent: float,
                 score_percent: float,
                 cl_scale:float,
                 bert_name: str,
                 beta:float):
        super(PhraseClassifier, self).__init__()

        self._lexical_vocab = lexical_vocab
        self._label_vocab = label_vocab
        self._neg_rate = neg_rate
        self._clloss_percent = clloss_percent
        self._score_percent = score_percent
        self._cl_scale = cl_scale
        self._beta = beta
        self._encoder = BERT(bert_name)
        self._classifier = MLP(self._encoder.dimension * 4, hidden_dim, len(label_vocab), dropout_rate)
        self._criterion = nn.NLLLoss()
        self._SCL = SupConLossPLMS(temperature=0.05)
        self.p_norm = pNorm()
        self._OGCE = GCELoss(num_classes=len(label_vocab), q=0.3)
        self._Focalloss = FocalLoss(gamma=2, alpha=0.5, num_classes=len(label_vocab), beta=self._beta)
        self.relative_positions_encoding = relative_position_encoding(max_length=512,
                                                                      depth= 4*self._encoder.dimension,
                                                                      max_relative_position=127)

    def forward(self, var_h, **kwargs):
        con_repr = self._encoder(var_h, kwargs["mask_mat"], kwargs["starts"])

        batch_size, token_num, hidden_dim = con_repr.size()
        ext_row = con_repr.unsqueeze(2).expand(batch_size, token_num, token_num, hidden_dim)
        ext_column = con_repr.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_column, ext_row - ext_column, ext_row * ext_column], dim=-1)

        relations_keys = self.relative_positions_encoding[:token_num, :token_num, :].cuda()
        table += relations_keys

        return self._classifier(table), self._classifier.get_dense(table)

    def _pre_process_input(self, utterances):
        lengths = [len(s) for s in utterances]
        max_len = max(lengths)
        pieces = iterative_support(self._lexical_vocab.tokenize, utterances)
        units, positions = [], []

        for tokens in pieces:
            units.append(flat_list(tokens))
            cum_list = np.cumsum([len(p) for p in tokens]).tolist()
            positions.append([0] + cum_list[:-1])

        sizes = [len(u) for u in units]
        max_size = max(sizes)
        cls_sign = self._lexical_vocab.CLS_SIGN
        sep_sign = self._lexical_vocab.SEP_SIGN
        pad_sign = self._lexical_vocab.PAD_SIGN
        pad_unit = [[cls_sign] + s + [sep_sign] + [pad_sign] * (max_size - len(s)) for s in units]
        starts = [[ln + 1 for ln in u] + [max_size + 1] * (max_len - len(u)) for u in positions]

        var_unit = torch.LongTensor([self._lexical_vocab.index(u) for u in pad_unit])
        attn_mask = torch.LongTensor([[1] * (lg + 2) + [0] * (max_size - lg) for lg in sizes])
        var_start = torch.LongTensor(starts)

        if torch.cuda.is_available():
            var_unit = var_unit.cuda()
            attn_mask = attn_mask.cuda()
            var_start = var_start.cuda()
        return var_unit, attn_mask, var_start, lengths

    def knn_inference_score(self, dict_knn, embedding_t, score_t, bz, len_1, len_2, dim, k=128, hard_score=True):

        knn_tensor_all = []
        knn_label_all = []

        for tensor_index, tensor_value in dict_knn.items():
            tensor_label = tensor_value[1][0]
            tensor_item = tensor_value[0][0]
            knn_label_all.append(tensor_label)
            knn_tensor_all.append(tensor_item)

        knn_label_all = np.array(knn_label_all)

        center_tensor = torch.stack(knn_tensor_all)


        index_faiss = faiss.IndexFlatIP(dim)
        db_tensor = center_tensor.cpu().numpy()
        query_tensor = embedding_t.cpu().numpy()
        faiss.normalize_L2(db_tensor)
        faiss.normalize_L2(query_tensor)
        index_faiss.add(db_tensor)
        Dref, Iref = index_faiss.search(query_tensor, k)


        onehot_label = torch.tensor(knn_label_all[Iref])
        onehot_label = F.one_hot(onehot_label, num_classes=len(self._label_vocab))
        init_knn_sim = torch.sum(onehot_label, 1)

        if hard_score == True:
            mask = (init_knn_sim == init_knn_sim.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
            init_knn_sim = torch.mul(mask, init_knn_sim)
            ones = torch.ones_like(init_knn_sim)
            init_knn_sim = torch.where(init_knn_sim>0, ones, init_knn_sim)
        else:
            pass

        knn_score = init_knn_sim if hard_score else torch.softmax(init_knn_sim, dim=-1)
        knn_score = knn_score.view(bz, len_1, len_2, -1)

        return knn_score

    def _pre_process_output(self, entities, lengths):
        positions, labels = [], []
        batch_size = len(entities)

        for utt_i in range(0, batch_size):
            for segment in entities[utt_i]:
                positions.append((utt_i, segment[0], segment[1]))
                labels.append(segment[2])

        for utt_i in range(0, batch_size):
            reject_set = [(e[0], e[1]) for e in entities[utt_i]]
            s_len = lengths[utt_i]
            neg_num = int(s_len * self._neg_rate) + 1

            candies = flat_list([[(i, j) for j in range(i, s_len) if (i, j) not in reject_set] for i in range(s_len)])
            if len(candies) > 0:
                sample_num = min(neg_num, len(candies))
                assert sample_num > 0

                np.random.shuffle(candies)
                for i, j in candies[:sample_num]:
                    positions.append((utt_i, i, j))
                    labels.append("O")

        var_lbl = torch.LongTensor(iterative_support(self._label_vocab.index, labels))
        if torch.cuda.is_available():
            var_lbl = var_lbl.cuda()
        return positions, var_lbl

    def PU_mixup(self, flat_s, flat_e, targets, softlabel_matrix, threshold_min=0.35, threshold_max=0.65, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        flat_s = torch.softmax(flat_s, dim=-1)

        entity_index_set_list = np.argwhere(targets.cpu() != self._label_vocab.index('O')).reshape(-1).tolist()

        unlabel_index_set = []
        reliable_entity_set = []
        for i in range(0,flat_s.size()[0]):
            if (threshold_min < flat_s[i][self._label_vocab.index('O')]) and (flat_s[i][self._label_vocab.index('O')] < threshold_max) and (flat_s[i][self._label_vocab.index('O')] == flat_s[i].max()):
                unlabel_index_set.append(i)

                sort_s, idx1 = torch.sort(flat_s[i], descending=True)
                idx1 = idx1.tolist()
                idx_reliable_entity_label = idx1[0] if idx1[0] != self._label_vocab.index('O') else idx1[1]
                reliable_entity_index_set = np.argwhere(targets.cpu() == idx_reliable_entity_label).reshape(-1).tolist()

                if len(entity_index_set_list) == 0:
                    continue
                else:
                    entity_idex_index = random.sample(reliable_entity_index_set, 1)[0] if len(reliable_entity_index_set)>0 else random.sample(entity_index_set_list, 1)[0]
                    reliable_entity_set.append(entity_idex_index)


        if unlabel_index_set != [] and reliable_entity_set != []:

            unlabel_index_set = torch.tensor(unlabel_index_set)

            entity_match_list = torch.tensor(reliable_entity_set).cuda()

            entity_e = flat_e[entity_match_list]
            unlabel_e = flat_e[unlabel_index_set]
            entity_targets = targets[entity_match_list]
            unlabel_targets = targets[unlabel_index_set]
            mixup_e = lam * entity_e + (1-lam) * unlabel_e
            mixup_s = self._classifier.get_score(mixup_e)
            entity_CE_loss = self.apply_different_loss(mixup_s.cuda(), entity_targets.cuda(), softlabel_matrix)
            unlabel_CE_loss = calculate_pNorm_loss(self._OGCE, mixup_s.cuda(), unlabel_targets.cuda(), self.p_norm)
            loss = lam * entity_CE_loss + (1-lam)*unlabel_CE_loss
            loss = (entity_match_list.size()[0] / (targets.size()[0] + entity_match_list.size()[0])) * loss
            percent = 1 - (entity_match_list.size()[0] / (targets.size()[0]+ entity_match_list.size()[0]))
        else:
            loss = 0
            percent = 1
        return loss, percent

    def mixup(self, flat_s, flat_e, targets, softlabel_matrix, threshold_min=0.35, threshold_max=0.65, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        flat_s = torch.softmax(flat_s, dim=-1)


        unlabel_index_set = []
        reliable_entity_set = []
        for i in range(0,flat_s.size()[0]):
            unlabel_index_set.append(i)
        reliable_entity_set = unlabel_index_set
        random.shuffle(reliable_entity_set)

        if unlabel_index_set != [] and reliable_entity_set != []:

            unlabel_index_set = torch.tensor(unlabel_index_set)

            entity_match_list = torch.tensor(reliable_entity_set).cuda()

            entity_e = flat_e[entity_match_list]
            unlabel_e = flat_e[unlabel_index_set]
            entity_targets = targets[entity_match_list]
            unlabel_targets = targets[unlabel_index_set]
            mixup_e = lam * entity_e + (1-lam) * unlabel_e
            mixup_s = self._classifier.get_score(mixup_e)
            entity_CE_loss = self.apply_different_loss(mixup_s.cuda(), entity_targets.cuda(), softlabel_matrix)
            unlabel_CE_loss = calculate_pNorm_loss(self._OGCE, mixup_s.cuda(), unlabel_targets.cuda(), self.p_norm)
            loss = lam * entity_CE_loss + (1-lam)*unlabel_CE_loss
            loss = (entity_match_list.size()[0] / (targets.size()[0] + entity_match_list.size()[0])) * loss
            percent = 0.5
        else:
            loss = 0
            percent = 1
        return loss, percent

    def apply_different_loss(self, flat_s, targets, softlabel_matrix):
        entity_index_set = np.argwhere(targets.cpu() != self._label_vocab.index('O')).reshape(-1).cuda()
        O_index_set = np.argwhere(targets.cpu() == self._label_vocab.index('O')).reshape(-1).cuda()


        entity_targets = targets[entity_index_set]
        O_targets = targets[O_index_set]
        entity_s = flat_s[entity_index_set]

        entity_s = torch.log_softmax(entity_s, dim=-1)
        entity_loss = self._Focalloss(entity_s,entity_targets.cuda(),softlabel_matrix)
        O_s = flat_s[O_index_set]


        O_loss = calculate_pNorm_loss(self._OGCE, O_s.cuda(), O_targets.cuda(), self.p_norm)

        loss = (len(entity_targets)/len(targets)) * entity_loss + (len(O_targets)/len(targets)) * O_loss

        return loss

    def estimate_CL(self, sentences, segments, softlabel_matrix):
        var_sent, attn_mask, start_mat, lengths = self._pre_process_input(sentences)
        score_t, embedding_t = self(var_sent, mask_mat=attn_mask, starts=start_mat)

        positions, targets = self._pre_process_output(segments, lengths)
        targets = targets.cuda()
        flat_s = torch.cat([score_t[[i], j, k] for i, j, k in positions], dim=0).cuda()
        flat_e = torch.cat([embedding_t[[i], j, k] for i, j, k in positions], dim=0).cuda()
        softmax_score = torch.softmax(flat_s, dim=-1)

        CE_loss = self.apply_different_loss(flat_s, targets, softlabel_matrix)

        loss_mixup, percent_loss = self.PU_mixup(flat_s, flat_e, targets, softlabel_matrix, threshold_min=0.35, threshold_max=0.5, alpha=0.2)
        CE_loss = percent_loss * CE_loss + loss_mixup

        #mix-up
        # loss_mixup, percent_loss = self.mixup(flat_s, flat_e, targets, softlabel_matrix, threshold_min=0.35, threshold_max=0.5, alpha=0.2)
        # CE_loss = percent_loss * CE_loss + loss_mixup

        CL_loss = self._SCL(flat_e, targets.cuda(), len(self._label_vocab), self._label_vocab)
        CE_loss = (1 - self._clloss_percent) * CE_loss + (self._clloss_percent * CL_loss)

        dict_knn = {}
        target_num = len(flat_e)
        for i in range(0,target_num):
            dict_knn[i] = [[flat_e[i].detach().cpu()], [targets[i].item()]]
        return CE_loss, dict_knn, softmax_score, targets

    def inference(self, sentences, dict_knn, knn= False, theorhold=0, k=64):
        var_sent, attn_mask, starts, lengths = self._pre_process_input(sentences)
        log_items, embedding_t = self(var_sent, mask_mat=attn_mask, starts=starts)
        score_t = torch.log_softmax(log_items, dim=-1) #原本使用了logsoftmax
        bz, len_1 ,len_2 ,hidden_len = embedding_t.size()
        embedding_t = embedding_t.view(-1,hidden_len)


        if knn:
            knn_score = self.knn_inference_score(dict_knn, embedding_t, score_t, bz, len_1 ,len_2, dim=hidden_len, k=k, hard_score=True).cpu()
            score_result = (1-self._score_percent) * score_t.cpu() + self._score_percent * knn_score.cpu()
            theorhold_tensor = torch.ones(1, len(self._label_vocab)) * theorhold
            theorhold_tensor[0] = 0
            theorhold_tensor = theorhold_tensor.repeat(bz, len_1, len_2, 1).cpu()
            score_result = score_result - theorhold_tensor

        else:
            score_result = score_t.cpu()
            theorhold_tensor = torch.ones(1, len(self._label_vocab)) * theorhold
            score_result = score_result - theorhold_tensor



        val_table, idx_table = torch.max(score_result, dim=-1)
        listing_it = idx_table.cpu().numpy().tolist()
        listing_vt = val_table.cpu().numpy().tolist()
        label_table = iterative_support(self._label_vocab.get, listing_it)

        candidates = []
        for l_mat, v_mat, sent_l in zip(label_table, listing_vt, lengths):
            candidates.append([])
            for i in range(0, sent_l):
                for j in range(i, sent_l):
                    if l_mat[i][j] != "O":
                        candidates[-1].append((i, j, l_mat[i][j], v_mat[i][j]))

        entities = []
        for segments in candidates:
            ordered_seg = sorted(segments, key=lambda e: -e[-1])
            filter_list = []
            for elem in ordered_seg:
                flag = False
                current = (elem[0], elem[1])
                for prior in filter_list:
                    flag = conflict_judge(current, (prior[0], prior[1]))
                    if flag:
                        break
                if not flag:
                    filter_list.append((elem[0], elem[1], elem[2]))
            entities.append(sorted(filter_list, key=lambda e: e[0]))
        return entities

class BERT(nn.Module):

    def __init__(self, bert_name):
        super(BERT, self).__init__()
        if 'roberta' in bert_name:
            self.config = RobertaConfig.from_pretrained('resource/roberta-base', output_hidden_states=False)
            self._repr_model = RobertaModel.from_pretrained('resource/roberta-base', config=self.config)
        elif 'bio' in bert_name:
            self.config = AutoConfig.from_pretrained('resource/biobert-base-cased-v1.1', output_hidden_states=False)
            self._repr_model = AutoModel.from_pretrained('resource/biobert-base-cased-v1.1', config=self.config)
        elif 'chinese' in bert_name:
            self.config = AutoConfig.from_pretrained('resource/bert-base-chinese', output_hidden_states=False)
            self._repr_model = AutoModel.from_pretrained('resource/bert-base-chinese', config=self.config)
        else:
            self.config = AutoConfig.from_pretrained('resource/bert-base-cased', output_hidden_states = False)
            self._repr_model = AutoModel.from_pretrained('resource/bert-base-cased', config = self.config)

    @property
    def dimension(self):
        return 768

    def forward(self, var_h, attn_mask, starts):
        all_hidden, _ = self._repr_model(var_h, attention_mask=attn_mask, return_dict=False)
        batch_size, _, hidden_dim = all_hidden.size()
        _, unit_num = starts.size()
        positions = starts.unsqueeze(-1).expand(batch_size, unit_num, hidden_dim)
        return torch.gather(all_hidden, dim=-2, index=positions)

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()

        self._densenet = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                      nn.Tanh())
        self._scorer = nn.Linear(hidden_dim, output_dim)
        self._dropout = nn.Dropout(dropout_rate)


    def forward(self, var_h):
        return self._scorer(self._densenet(self._dropout(var_h)))

    def get_dense(self, var_h):
        return self._densenet(self._dropout(var_h))

    def get_score(self, dense):
        return self._scorer(dense)



