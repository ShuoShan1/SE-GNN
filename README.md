# SE-GNN: Seed Expanded-Aware Graph Neural Network with Iterative Optimization for Semi-supervised Entity Alignment



## Abstract

> Entity alignment aims to use pre-aligned seed pairs to find other equivalent entities from different knowledge graphs and is widely used in graph fusion-related fields. However, as the scale of knowledge graphs increases, manually annotating pre-aligned seed pairs becomes difficult. Existing research utilizes entity embeddings obtained by aggregating structural information to identify potential seed pairs, thus reducing the reliance on pre-aligned seed pairs. However, due to the structural heterogeneity of KG, the quality of potential seed pairs obtained using only structural information is not ideal. In addition, although existing research improves the quality of potential seed pairs through semi-supervised iteration, they underestimate the impact of embedding distortion produced by noisy seed pairs on the alignment effect. In order to solve the above problems, we propose a seed expanded-aware graph neural network with iterative optimization for semi-supervised entity alignment, named SE-GNN. First, we utilize the semantic attributes and structural features of entities, combined with a conditional filtering mechanism, to obtain high-quality initial potential seed pairs. Next, we designed a local and global awareness mechanism. It introduces initial potential seed pairs and combines local and global information to obtain a more comprehensive entity embedding representation, which alleviates the impact of KG structural heterogeneity and lays the foundation for the optimization of initial potential seed pairs. Then, we designed the threshold nearest neighbor embedding correction strategy. It combines the similarity threshold and the bidirectional nearest neighbor method as a filtering mechanism to select iterative potential seed pairs and also uses an embedding correction strategy to eliminate the embedding distortion. Finally, we will reach the optimized potential seeds after iterative rounds to input local and global sensing mechanisms, obtain the final entity embedding, and perform entity alignment. Experimental results on public datasets demonstrate the excellent performance of our SE-GNN, showcasing the effectiveness of the model.



## Datasets



DBP15K(ZH-EN),  DBP15K(FR-EN),  DBP15K(JA-EN);

SRPRS(EN-FR),SRPRS(EN-DE)



## Environment

The essential packages to run the code:

- python3.7
- pytorch
- numpy
- torch-scatter 
- scipy
- tabulate

