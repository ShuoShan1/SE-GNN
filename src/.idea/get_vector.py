import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
#from langchain.embeddings import HuggingFaceBgeEmbeddings
from torch.nn import CosineSimilarity

# EMBEDDING_DEVICE = "cuda:0" if torch.cuda.is_available(
# ) else "mps" if torch.backends.mps.is_available() else "cpu"
# model_kwargs = {'device': "cpu"}

# embedding_model_dict = {
#     "text2vec-base": "models/text2vec-base-chinese",
#     "text2vec-large": "models/text2vec-large-chinese",
#     "m3e-base": "models/m3e-base",
#     "bge-large": "models/bge-large-zh",
#     "bge-large-en": "E:\ss\\attr-seedpair-iterate\\bge-large-en"
# }


model_name = "E:\ss\\attr-seedpair-iterate\models\\bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceEmbeddings(
    model_name=model_name
)

# 获取嵌入模型
# def get_embedding(model_name="bge-large-en"):
#     embedding = HuggingFaceEmbeddings(
#         model_name=embedding_model_dict[model_name], model_kwargs=model_kwargs)
#     print("获取的嵌入模型是:", model_name)
#     return embedding


# 获取余弦相似度函数，torch.nn-i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# def get_simfunc():
#     simfunc = CosineSimilarity(dim=0, eps=1e-6)
#     return simfunc


if __name__ == '__main__':
    titles = ["苹果", "Apple","华为"]
    embedding = model(model_name="bge-large-en")
    t_1 = torch.tensor(embedding.embed_query(titles[0]), dtype=float)
    t_2 = torch.tensor(embedding.embed_query(titles[1]), dtype=float)
    # score = get_simfunc(t_1, t_2)
    tool = get_simfunc()
    score = tool(t_1, t_2)
    print(score)
