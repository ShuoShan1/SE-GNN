import logging
import torch
from datetime import datetime
import torch
import pickle
import numpy as np


def set_device(args):
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    if device == 'cpu':
        logging.info("\n no gpu found, program is running on cpu! \n")
    return device

def get_cosine_matrix2(lemb, remb):#多个向量求余弦
    lemb = lemb / (torch.linalg.norm(lemb, dim=-1, keepdim=True) + 1e-5)
    remb = remb / (torch.linalg.norm(remb, dim=-1, keepdim=True) + 1e-5)
    A_sim = torch.mm(lemb, remb.t())
    return A_sim

def load_entityid(filename):
    entityid = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            index = line.split()[0]
            entityid.append(index)
    entityid = list(map(int, entityid))
    return entityid

def get_cosine_matrix1(lemb, remb):
    lemb = torch.stack(tuple(lemb))
    remb = torch.stack(tuple(remb))
    lemb = lemb / (torch.linalg.norm(lemb, dim=-1, keepdim=True) + 1e-5)
    remb = remb / (torch.linalg.norm(remb, dim=-1, keepdim=True) + 1e-5)
    A_sim = torch.mm(lemb, remb.t())
    B_sim = torch.mm(remb, lemb.t())
    k = 10
    A_avg = torch.sum(torch.topk(A_sim, k=k)[0], dim=-1) / k
    B_avg = torch.sum(torch.topk(B_sim, k=k)[0], dim=-1) / k
    B_avg, A_avg = [torch.unsqueeze(m, dim=1) for m in [B_avg, A_avg]]
    cosine_matrix = 2 * A_sim - A_avg - B_avg.t()
    return cosine_matrix

def get_newpairs(cosine,entityid1,entityid2,threshold):
    new_seedpairs = []
    max_values, max_indices = torch.max(cosine, dim=1)
    for i in range(len(entityid1)):
        if max_values[i] > threshold:
            max_similarity_pair = [entityid1[i], entityid2[max_indices[i]]]
            new_seedpairs.append(max_similarity_pair)
    return new_seedpairs

def generate_matching_pairs(new_seedpairs1, new_seedpairs2):#互相最置信的保存
    new_seedpairs = []
    for seedpairs1 in new_seedpairs1:
        for seedpairs2 in new_seedpairs2:
            if seedpairs1[0] == seedpairs2[1] and seedpairs1[1] == seedpairs2[0]:
                new_seedpairs.append(seedpairs1)
                break
    return new_seedpairs

# def get_pair(train_pair,lan):
#
#     google_id = load_entityid("../datasets/DBP15K/" + lan + "_en/ent_ids_1")
#     en_id = load_entityid("../datasets/DBP15K/" + lan + "_en/ent_ids_2")
#     with open("../entity_emb/" + lan + "_google.pickle", 'rb') as file:
#         google_emb = pickle.load(file)
#     with open("../entity_emb/" + lan + "_en.pickle", 'rb') as file:
#         en_emb = pickle.load(file)
#
#     google_emb1 = torch.zeros((len(google_id), 1024))
#     en_emb1 = torch.zeros((len(en_id), 1024))
#     for i in range(len(google_id)):
#         google_emb1[i] = google_emb[i]
#     for i in range(len(en_id)):
#         en_emb1[i] = en_emb[i]
#
#     cosine1 = get_cosine_matrix1(google_emb1, en_emb1)
#     cosine2 = get_cosine_matrix1(en_emb1, google_emb1)
#
#     new_seedpair1 = get_newpairs(cosine1, google_id, en_id, 0.05)
#     new_seedpair2 = get_newpairs(cosine2, en_id, google_id, 0.05)
#     new_seedpair = generate_matching_pairs(new_seedpair1, new_seedpair2)
#     new_seedpairs = []
#     #
#
#     for seedpair in new_seedpair:
#         if not any(all(element in a_list for element in seedpair) for a_list in train_pair):
#             new_seedpairs.append(seedpair)
#
#     return new_seedpairs




def high_nei(lan):
    google_id = load_entityid("../datasets/DBP15K/" + lan + "_en/ent_ids_1")
    en_id = load_entityid("../datasets/DBP15K/" + lan + "_en/ent_ids_2")
    with open("../entity_emb/" + lan + "_google.pickle", 'rb') as file:
        google_emb = pickle.load(file)
    with open("../entity_emb/" + lan + "_en.pickle", 'rb') as file:
        en_emb = pickle.load(file)

    zh_google_emb1 = torch.zeros((len(google_id), 1024))
    en_emb1 = torch.zeros((len(en_id), 1024))
    for i in range(len(google_id)):
        zh_google_emb1[i] = google_emb[i]
    for i in range(len(en_id)):
        en_emb1[i] = en_emb[i]
    entity_num = len(google_id) + len(en_id)
    cosine1 = get_cosine_matrix2(zh_google_emb1, zh_google_emb1)
    cosine2 = get_cosine_matrix2(en_emb1, en_emb1)
    top_values1, top_indices1 = torch.topk(cosine1, k=35, dim=1)
    top_values2, top_indices2 = torch.topk(cosine2, k=35, dim=1)
    K = 15
    high_adj = torch.zeros((entity_num, entity_num),dtype=int)
    top_id1 = torch.zeros((len(cosine1), K))
    top_id2 = torch.zeros((len(cosine2), K))
    for i in range(len(cosine1)):
        for j in range(K):
            top_id1[i][j] = google_id[top_indices1[i][j]]
            high_adj[google_id[i]][int(top_id1[i][j])] = 1
    for i in range(len(cosine2)):
        for j in range(K):
            top_id2[i][j] = en_id[top_indices2[i][j]]
            high_adj[en_id[i]][int(top_id2[i][j])] = 1

    high_adj = np.stack(high_adj.nonzero(), axis=1)
    high_adj = torch.from_numpy(np.transpose(high_adj))
    print("-----------Electing completed-----------")
    return high_adj.T


def get_pair_nei(train_pair,adj,lan):
    print("--Load BGE embed--")
    google_id = load_entityid("../datasets/DBP15K/" + lan + "_en/ent_ids_1")
    en_id = load_entityid("../datasets/DBP15K/" + lan + "_en/ent_ids_2")
    with open("../entity_emb/" + lan + "_google.pickle", 'rb') as file:
        google_emb = pickle.load(file)
    with open("../entity_emb/" + lan + "_en.pickle", 'rb') as file:
        en_emb = pickle.load(file)
    a = 0.5

    # Get the similarity matrix of entity embedding
    print("--Build matrix--")
    zh_google_emb1 = torch.zeros((len(google_id), 1024))
    en_emb1 = torch.zeros((len(en_id), 1024))
    for i in range(len(google_id)):
        zh_google_emb1[i] = google_emb[i]
    for i in range(len(en_id)):
        en_emb1[i] = en_emb[i]
    cosine1 = get_cosine_matrix1(zh_google_emb1, en_emb1)
    cosine2 = get_cosine_matrix1(en_emb1, zh_google_emb1)

    #Get the similarity matrix of neighborhood entity embeddings
    entity_num = len(google_id) + len(en_id)
    shape = (entity_num, 1024)
    entity_emb = torch.zeros(shape)

    for i in range(len(google_id)):
        index = google_id[i]
        entity_emb[index] = google_emb[i]
    for i in range(len(en_id)):
        index = en_id[i]
        entity_emb[index] = en_emb[i]
    adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),size=[entity_num, entity_num])
    adj = torch.sparse.softmax(adj, dim=1)
    entity_emb = torch.sparse.mm(adj, entity_emb)

    google_emb2 = torch.zeros((len(google_id), 1024))
    en_emb2 = torch.zeros((len(en_id), 1024))
    for i,idx in enumerate(google_id):
        google_emb2[i] = entity_emb[idx]
    for i,idx in enumerate(en_id):
        en_emb2[i] = entity_emb[idx]
    cosine11 = get_cosine_matrix1(google_emb2, en_emb2)
    cosine22 = get_cosine_matrix1(en_emb2, google_emb2)

    #Similarity matrix fusion
    print("--Matrix fusion--")
    cosine111 = a * cosine1 + (1-a)*cosine11
    cosine222 = a * cosine2 + (1-a)*cosine22
    new_seedpair1 = get_newpairs(cosine111, google_id, en_id, 0.01)
    new_seedpair2 = get_newpairs(cosine222, en_id, google_id, 0.01)
    new_seedpair = generate_matching_pairs(new_seedpair1, new_seedpair2)
    new_seedpairs = []

    # Eliminate pre-aligned seeds
    print("--Eliminate--")
    for seedpair in new_seedpair:
        if not any(all(element in a_list for element in seedpair) for a_list in train_pair):
            new_seedpairs.append(seedpair)

    return new_seedpairs







