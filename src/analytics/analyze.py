# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

from gensim import corpora
from gensim.corpora import Dictionary

from src.analytics.nlp_utils import *
from src.analytics.ml_utils import *
import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Analyzer:
    """
    """
    def __init__(self):
        """
        """

    def get_clusters(self, embed_vectors, cluster_no: int = 10, with_tsne: bool = False):
        """
        :param embed_vectors:
        :param cluster_no:
        :param with_tsne:
        :return:
        """
        if with_tsne:
            cluster_pred = run_kmeans(embed_vectors, k=cluster_no)
            tsne_vectors = compute_tsne(numpy.array(embed_vectors))
            return cluster_pred, tsne_vectors
        else:
            cluster_pred = run_kmeans(embed_vectors, k=cluster_no)
            return cluster_pred


    def get_cluster_plot_with_tsne(self, tsne_vector, cluster_pred):
        """

        :param tsne_vector:
        :param cluster_pred:
        :return:
        """
        dftsne = pd.DataFrame(tsne_vector)
        dftsne['cluster'] = cluster_pred
        dftsne.columns = ['x1', 'x2', 'cluster']
        sns.set(rc={'figure.figsize': (10, 10)})
        palette = sns.color_palette("bright", len(set(cluster_pred)))

        ax2 = sns.scatterplot(data=dftsne, x='x1', y='x2', hue=cluster_pred, legend="full", palette=palette, alpha=0.5)
        plt.title("t-SNE Plasma related Patents - Clustered")
        return ax2


    def add_embedding_vector(self, df, bert_model_name:str='PATENT_BERT'):
        """
        add Embedding vector for each text in the df. The df should have 'text' field
        :param df:
        :param bert_model_name:
        :return:
        """
        df['embeddings'] = df['text'].map(lambda line: get_embeddings(line))

        return df






















