import pickle
import torch
import torch.nn as nn
from gnn_model import Encoder_Model
import numpy as np
import argparse
import time
from utils import *
from evals import evaluate
from data_loader import KGs
import logging
from seed_iterate import *



def CSLS_evaluate():
    Lvec, Rvec,out_feature = model.get_embeddings(test_pair[:, 0], test_pair[:, 1])
    evals = evaluate(dev_pair=test_pair)
    results = evals.CSLS_cal(Lvec, Rvec)
    def cal(results):
        hits1, hits10, mrr = 0, 0, 0
        for x in results[:, 1]:
            if x < 1:
                hits1 += 1
            if x < 10:
                hits10 += 1
            mrr += 1 / (x + 1)
        return hits1, hits10, mrr

    hits1, hits10, mrr = cal(results)
    print("Hits@1: ", hits1 / len(Lvec), " ",  "Hits@10: ", hits10 / len(Lvec)," ", "MRR: ", mrr / len(Lvec))
    return out_feature


def train_base(args,train_pairs,model:Encoder_Model):
    flag = 1
    total_train_time = 0.0
    for epoch in range(args.epoch):
        time1 = time.time()
        total_loss = 0
        np.random.shuffle(train_pairs)
        batch_num = len(train_pairs) // args.batch_size + 1
        model.train()
        for b in range(batch_num):
            pairs = train_pairs[b * args.batch_size:(b + 1) * args.batch_size]
            if len(pairs) == 0:
                continue
            pairs = torch.from_numpy(pairs).to(device)
            optimizer.zero_grad()
            loss = model(pairs, flag)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        time2 = time.time()
        total_train_time += time2 - time1
        print(f'[epoch {epoch + 1}/{args.epoch}]  epoch loss: {(total_loss):.5f}, time cost: {(time2 - time1):.3f}s')
        flag = 1

        round = 30
        # -----------Validation-----------
        if (epoch + 1) > 90 and (epoch + 1) % 5 == 0 or (epoch + 1) % round == 0 :
            print("-----------Validation-----------")
            model.eval()
            with torch.no_grad():
                out_feature = CSLS_evaluate()

        # -----------Iteration-----------
        if (epoch + 1) in [round, round*2, round*3]:
            print("-----------Start optimize potential seed pairs-----------")
            opt_seedpairs = bnns(entity1, entity2, out_feature, train_pair, 0.05)
            train_pairs = np.concatenate((train_pair, opt_seedpairs))
            flag = 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='alignment model')
    parser.add_argument('--log_path', default='../logs', type=str)
    parser.add_argument('--dataset', default='DBP15K', type=str)
    parser.add_argument('--batch', default='base', type=str)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--ent_hidden', default=100, type=int)
    parser.add_argument('--rel_hidden', default=100, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--ind_dropout_rate', default=0.3, type=float)

    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--gamma', default=2.0, type=float)

    parser.add_argument('--eval_batch_size', default=512, type=int)
    parser.add_argument('--dev_interval', default=2, type=int)
    parser.add_argument('--stop_step', default=3, type=int)
    parser.add_argument('--sim_threshold', default=0.0, type=float)
    parser.add_argument('--topk', default=1, type=int)
    parser.add_argument('--M', default=500, type=int)
    args = parser.parse_args()
    language = "zh"
    device = set_device(args)


    print("-----------load KG-----------")
    kgs = KGs()
    train_pair, valid_pair, test_pair, ent_adj, r_index, r_val, ent_adj_with_loop, ent_rel_adj,entity1,entity2,ill_ent,triples1,triples2 = kgs.load_data(args,language)
    ent_adj = torch.from_numpy(np.transpose(ent_adj))
    ent_rel_adj = torch.from_numpy(np.transpose(ent_rel_adj))
    ent_adj_with_loop = torch.from_numpy(np.transpose(ent_adj_with_loop))
    r_index = torch.from_numpy(np.transpose(r_index))
    r_val = torch.from_numpy(r_val)


    print("-----------Start seed expansion-----------")
    # new_pair = get_pair_nei(train_pair, ent_adj,language)
    with open("../entity_pairs/" + language + "_pair_0.01.pickle", 'rb') as file:
        new_pair = pickle.load(file)
    train_pairs = np.concatenate((train_pair, new_pair))

    # -----------Higher-order neighbors-----------
    print("-----------Electing higher-order neighbors-----------")
    high_adj= high_nei(language)


    model = Encoder_Model(node_hidden=args.ent_hidden,
                          rel_hidden=args.rel_hidden,
                          node_size=kgs.old_ent_num,
                          rel_size=kgs.total_rel_num,
                          triple_size=kgs.triple_num,
                          device=device,
                          adj_matrix=ent_adj,
                          r_index=r_index,
                          r_val=r_val,
                          rel_matrix=ent_rel_adj,
                          ent_matrix=ent_adj_with_loop,
                          ill_ent=ill_ent,
                          dropout_rate=args.dropout_rate,
                          gamma=args.gamma,
                          lr=args.lr,
                          depth=args.depth,
                          high_adj = high_adj
                          ).to(device)


    evaluator = evaluate(dev_pair = test_pair)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    if 'base' in args.batch:
        train_base(args,train_pairs,model)

