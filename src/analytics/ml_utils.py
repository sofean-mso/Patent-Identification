# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)


from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


def compute_tfidf(corpus):
    """

    :param corpus:
    :return:
    """
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                        encoding='utf-8',
                        lowercase=True,
                        min_df=3,
                        max_df=0.9,
                        norm='l2',
                       token_pattern = '[a-zA-Z0-9_-]+',
                        ngram_range=(1, 2),
                        smooth_idf=True,
                        preprocessor=None,
                        max_features=5000)
    X = vectorizer.fit_transform(corpus)

    return X


def compute_tsne(corpus_vectors):
    """
    run TSNE algorithm
    :param corpus_vectors:
    :return:
    """
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, random_state=100, n_iter=5000)
    tsne_embeddings = tsne.fit_transform(corpus_vectors)

    return tsne_embeddings


def run_kmeans(corpus_vectors, k: int = 10, max_itr: int = 3000):
    """

    :param corpus_vectors:
    :param k:
    :param max_itr:
    :return:
    """
    """
    run K-means algorithm
    :param corpus_vectors:
    :param k:
    :param max_itr:
    :return:
    """
    kmeans = KMeans(n_clusters=k, verbose=10, max_iter=max_itr)
    y_pred = kmeans.fit_predict(corpus_vectors)

    return y_pred


def compute_elbow_method(corpus_vectors, min_k:int=3, max_k:int = 10):
    """

    :param corpus_vectors:
    :param min_k:
    :param max_k:
    :return:
    """
    Sum_of_squared_distances = []
    K = range(min_k, max_k)
    for k in K:
        km = KMeans(init="k-means++", n_clusters=k)
        km = km.fit(corpus_vectors)
        Sum_of_squared_distances.append(km.inertia_)
        print(' (Elbow) k-means für K ', k)

    ax = sns.lineplot(x=K, y=Sum_of_squared_distances)
    ax.lines[0].set_linestyle("--")

    plt.xlabel('k')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method For Optimal k')

    return ax


