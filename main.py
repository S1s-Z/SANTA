import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import json
import pickle
import torch
from transformers import AdamW,get_linear_schedule_with_warmup

from utils import UnitAlphabet, LabelAlphabet
from model import PhraseClassifier
from misc import fix_random_seed
from utils import corpus_to_iterator, Procedure

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", "-mode", type=str, required=True, default='train')
    parser.add_argument("--data_dir", "-dd", type=str, required=True)
    parser.add_argument("--check_dir", "-cd", type=str, required=True)
    parser.add_argument("--resource_dir", "-rd", type=str, required=True)
    parser.add_argument("--random_state", "-rs", type=int, default=0)
    parser.add_argument("--epoch_num", "-en", type=int, default=40)
    parser.add_argument("--batch_size", "-bs", type=int, default=16)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--negative_rate", "-nr", type=float, default=0.35)
    parser.add_argument("--warmup_proportion", "-wp", type=float, default=0.1)
    parser.add_argument("--pretrain_model", "-pm", type=str, default='bert')
    parser.add_argument("--hidden_dim", "-hd", type=int, default=256)
    parser.add_argument("--dropout_rate", "-dr", type=float, default=0.4)
    parser.add_argument("--CLloss_percent", "-lp", type=float, default=0.1)
    parser.add_argument("--score_percent", "-sp", type=float, default=0.7)
    parser.add_argument("--knn", "-knn", type=bool, default=False)
    parser.add_argument("--theorhold", "-th", type=float, default=0)
    parser.add_argument("--beta", "-beta", type=float, default=0.5)
    parser.add_argument("--cl_scale", "-cs", type=int, default=1)
    parser.add_argument("--k", "-k", type=int, default=64)


    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=True), end="\n\n")

    fix_random_seed(args.random_state)
    bert_name = args.pretrain_model
    lexical_vocab = UnitAlphabet(bert_name)
    label_vocab = LabelAlphabet()

    if args.mode == 'train':
        train_loader = corpus_to_iterator(os.path.join(args.data_dir, "train.json"), args.batch_size, True, label_vocab, False)
    dev_loader = corpus_to_iterator(os.path.join(args.data_dir, "dev.json"), args.batch_size, False)
    test_loader = corpus_to_iterator(os.path.join(args.data_dir, "test.json"), args.batch_size, False)

    model = PhraseClassifier(lexical_vocab, label_vocab, args.hidden_dim,
                             args.dropout_rate, args.negative_rate,
                             args.CLloss_percent, args.score_percent,
                             args.cl_scale, #args.cl_temp, args.use_detach,
                             bert_name, beta=args.beta)

    model = model.cuda() if torch.cuda.is_available() else model.cpu()

    all_parameters = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_param = [{'params': [p for n, p in all_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                     {'params': [p for n, p in all_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.00}]
    if args.mode == 'train':
        total_steps = int(len(train_loader) * (args.epoch_num + 1))
        optimizer = AdamW(grouped_param, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_proportion * total_steps , num_training_steps = total_steps)
    if not os.path.exists(args.check_dir):
        os.makedirs(args.check_dir)
    best_dev = 0.0
    best_test = 0.0
    best_precision = 0.0
    best_recall = 0.0
    script_path = os.path.join(args.resource_dir, "conlleval.pl")
    checkpoint_path = os.path.join(args.check_dir, "model.pt")

    softlabel_matrix = torch.eye(len(label_vocab)).cuda()
    list_softlabel_matrix = []
    list_softlabel_matrix.append(softlabel_matrix)
    if args.mode == 'train':
        for epoch_i in range(0, args.epoch_num + 1):
            loss, train_time, dict_center, softlabel_matrix_new = Procedure.train(model, train_loader, optimizer,scheduler, label_vocab, softlabel_matrix)

            print("[Epoch {:3d}] uses {:.3f} secs".format(epoch_i, train_time))
            softlabel_matrix = softlabel_matrix_new

            test_f1, test_precision, test_recall, test_time = Procedure.test(model, test_loader, script_path, dict_center, args.knn, args.theorhold, args.k)
            print("{{Epoch {:3d}}} f1 score on test set is {:.5f}, precision is {:.5f}, recall is {:.5f}, using {:.3f} secs".format(epoch_i, test_f1, test_precision,test_recall,test_time))

            if test_f1 > best_test:
                best_test = test_f1
                best_precision = test_precision
                best_recall = test_recall
                print("\nSave best test model with score: {:.5f} in terms of test set".format(test_f1))
                # torch.save(model, checkpoint_path)

                dict_path = os.path.join(args.check_dir, "dict_file.pkl")
                dict_save = open(dict_path, 'wb')
                pickle.dump(dict_center, dict_save)
                dict_save.close()

            print("\nbest test f1 score: {:.5f}. precision is {:.5f} and recall is {:.5f}".format(best_test,best_precision,best_recall))
            print(end="\n\n")
    else:
        dict_path = os.path.join(args.check_dir, "dict_file.pkl")
        dict_save = open(dict_path, 'rb')
        dict_center = pickle.load(dict_save)
        dict_save.close()
        model = torch.load(checkpoint_path)
        test_f1, test_precision, test_recall, test_time = Procedure.test(model, test_loader, script_path, dict_center, args.knn, args.theorhold, args.k)
        print("f1 score on test set is {:.5f}, precision is {:.5f}, recall is {:.5f}, using {:.3f} secs".format(test_f1, test_precision, test_recall, test_time))

