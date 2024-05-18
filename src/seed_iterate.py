from utils import *
from tqdm import tqdm

def get_cosine_matrix(entity1,entity2,emb):
    lemb, remb, cosine_matrix= [], [], []
    for idx in entity1:
        lemb.append(emb[idx])
    for idx in entity2:
        remb.append(emb[idx])
    lemb = torch.stack(lemb)
    remb = torch.stack(remb)
    A_sim = torch.mm(lemb, remb.t())
    B_sim = torch.mm(remb, lemb.t())
    k = 10
    A_avg = torch.sum(torch.topk(A_sim, k=k)[0], dim=-1) / k
    B_avg = torch.sum(torch.topk(B_sim, k=k)[0], dim=-1) / k
    B_avg, A_avg = [torch.unsqueeze(m, dim=1) for m in [B_avg, A_avg]]
    cosine_matrix = (2 * A_sim - A_avg - B_avg.t())/2
    return cosine_matrix.numpy()


def generate_matching_pairs(new_seedpairs1, new_seedpairs2):
    new_seedpairs = []
    for seedpairs1 in new_seedpairs1:
        for seedpairs2 in new_seedpairs2:
            if seedpairs1[0] == seedpairs2[1] and seedpairs1[1] == seedpairs2[0]:
                new_seedpairs.append(seedpairs1)
                break
    return new_seedpairs


def bnns(entity1,entity2,emb,train_pair,value):

    cosine1 = get_cosine_matrix(entity1, entity2, emb)
    cosine2 = get_cosine_matrix(entity2, entity1, emb)
    new_seedpair1 = get_newpairs(torch.from_numpy(cosine1), entity1, entity2, value)
    new_seedpair2 = get_newpairs(torch.from_numpy(cosine2), entity2, entity1, value)
    new_seedpair = generate_matching_pairs(new_seedpair1, new_seedpair2)
    new_seedpairs = []
    #
    for seedpair in tqdm(new_seedpair):
        if not any(all(element in a_list for element in seedpair) for a_list in train_pair):
            new_seedpairs.append(seedpair)
    print("-----------Optimize completed-----------")
    return new_seedpairs

