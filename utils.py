import numpy as np
from tqdm import tqdm
import time
import random

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer, pipeline

from misc import extract_json_data
from misc import iob_tagging, f1_score
import copy

class UnitAlphabet(object):

    CLS_SIGN, SEP_SIGN = "[CLS]", "[SEP]"
    PAD_SIGN, UNK_SIGN = "[PAD]", "[UNK]"

    def __init__(self, bert_name):
        if 'roberta' in bert_name:
            self._tokenizer = RobertaTokenizer.from_pretrained('resource/roberta-base', do_lower_case=False)
        elif 'bio' in bert_name:
            self._tokenizer = AutoTokenizer.from_pretrained('resource/biobert-base-cased-v1.1')
        elif 'chinese' in bert_name:
            self._tokenizer = AutoTokenizer.from_pretrained('resource/bert-base-chinese')
        else:
            self._tokenizer = AutoTokenizer.from_pretrained('resource/bert-base-cased',)
    def tokenize(self, item):
        return self._tokenizer.tokenize(item)

    def index(self, items):
        return self._tokenizer.convert_tokens_to_ids(items)


class LabelAlphabet(object):

    def __init__(self):
        super(LabelAlphabet, self).__init__()

        self._idx_to_item = []
        self._item_to_idx = {}

    def add(self, item):
        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def get(self, idx):
        return self._idx_to_item[idx]

    def index(self, item):
        return self._item_to_idx[item]

    def __str__(self):
        return str(self._item_to_idx)

    def __len__(self):
        return len(self._idx_to_item)


def corpus_to_iterator(file_path, batch_size, if_shuffle, label_vocab=None, AUG=False):
    material = extract_json_data(file_path)
    instances = [(eval(e["sentence"]), eval(e["labeled entities"])) for e in material]

    if label_vocab is not None:
        label_vocab.add("O")
        for _, u in instances:
            for _, _, l in u:
                label_vocab.add(l)

    class _DataSet(Dataset):

        def __init__(self, elements):
            self._elements = elements

        def __getitem__(self, item):
            return self._elements[item]

        def __len__(self):
            return len(self._elements)

    def distribute(elements):
        sentences, entities = [], []
        for s, e in elements:
            sentences.append(s)
            entities.append(e)
        return sentences, entities

    wrap_data = _DataSet(instances)
    return DataLoader(wrap_data, batch_size, if_shuffle, collate_fn=distribute)


class Procedure(object):

    @staticmethod
    def train(model, dataset, optimizer,scheduler, label_vocab, softlabel_matrix):
        model.train()
        time_start, total_penalties = time.time(), 0.0
        dict_result = {}
        flag_num = 0
        dict_index = 0
        entity_num = 0
        O_num = 0

        for batch in tqdm(dataset, ncols=50):
            loss, dict_knn, softmax_score, target_s = model.estimate_CL(*batch, softlabel_matrix)

            softlabel_matrix_new = torch.eye(len(label_vocab)).cuda()
            softlabel_matrix_new = torch.zeros_like(softlabel_matrix_new)
            for index_instance in range(0, softmax_score.size()[0]):
                if torch.max(softmax_score[index_instance], 0)[1] == target_s[index_instance]:
                    softlabel_matrix_new[target_s[index_instance]] = softlabel_matrix_new[target_s[index_instance]] + softmax_score[index_instance]
                else:
                    pass
            softlabel_matrix_new = torch.softmax(softlabel_matrix_new, dim=1)


            for tensor_index, tensor_value in dict_knn.items():
                tensor_label = tensor_value[1][0]
                O_index = label_vocab.index('O')
                if tensor_label == O_index:
                    O_num = O_num + 1
                else:
                    entity_num = entity_num + 1

            random.seed(1024)
            pick_list = random.sample(range(0, O_num), int(O_num * 0.00))

            O_put_into_dict_num = 0
            for tensor_index, tensor_value in dict_knn.items():
                tensor_label = tensor_value[1][0]
                O_index = label_vocab.index('O')
                if tensor_label != O_index:
                    dict_result[dict_index] = tensor_value
                    dict_index = dict_index + 1
                else:
                    pass
                    if O_put_into_dict_num in pick_list:
                        dict_result[dict_index] = tensor_value
                        dict_index = dict_index + 1
                    else:
                        pass
                    O_put_into_dict_num = O_put_into_dict_num + 1

            flag_num = flag_num + 1

            total_penalties += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5) # ori
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # onto
            optimizer.step()
            scheduler.step()


        time_con = time.time() - time_start
        return total_penalties, time_con, dict_result, softlabel_matrix_new

    @staticmethod
    def test(model, dataset, eval_path, dict_center, knn=False, theorhold=0 ,k=64):
        model.eval()
        time_start = time.time()
        seqs, outputs, oracles = [], [], []

        for sentences, segments in tqdm(dataset, ncols=50):
            with torch.no_grad():
                predictions = model.inference(sentences, dict_center, knn, theorhold, k)

            seqs.extend(sentences)
            outputs.extend([iob_tagging(e, len(u)) for e, u in zip(predictions, sentences)])
            oracles.extend([iob_tagging(e, len(u)) for e, u in zip(segments, sentences)])

        out_f1, out_precision, out_recall = f1_score(seqs, outputs, oracles, eval_path)
        return out_f1, out_precision, out_recall, time.time() - time_start
