import numpy as np

from src.analytics.analyze import  Analyzer
from src.analytics.preprocess import *

import pandas as pd
from src.analytics.ml_utils import *
from src.analytics.nlp_utils import *

import ast
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import seaborn as sns
s = "a plasma Physic is excitation. a plasma Physic of manufacturing an aperture plate using a plasma excitation chemical vapor deposition (CVD) device includes the steps of placing a metal plate in a vacuum chamber of the CVD device; discharging air inside the vacuum chamber; charging a mixture of a gas containing at least osmium and a gas containing a hydrogen gas; adjusting a pressure of the vacuum chamber at a predetermined level; and generating plasma inside the vacuum chamber. An electrically conductive amorphous coating having a dense structure is uniformly formed over a surface and an interior of a micro-hole of the aperture plate. Also, it is possible to form an osmium coating having a high purity and a low impurity content with good repeatability."

"""
ab_embedd = df['EMBEDDING'].tolist()
#embed_vectors = []
#for embed_vec in ab_embedd:
    #embed_vectors.append(ast.literal_eval(embed_vec))

embedd_vectors = [ast.literal_eval(embed_vec) for embed_vec in ab_embedd]


cluster_pred, tsne_vectors = analyzer.get_clusters(embedd_vectors, cluster_no=6, with_tsne=True)
ax = analyzer.get_cluster_plot_with_tsne(tsne_vector=tsne_vectors, cluster_pred=cluster_pred)
plt.show()


ax = compute_elbow_method(embedd_vectors, 3, 20)
plt.show()
"""


def show_tsne():
    #df = pd.read_csv("../data/2024/training/cleaned/tmp/all/plasma_training_dataset_0_1_SCIBERT_Embeddings.csv", encoding="utf-8", on_bad_lines='skip')

    df = pd.read_csv("../data/2024/training/cleaned/tmp/all/plasma_test_dataset_0_1._SCIBERT_Embeddings.csv",
                     encoding="utf-8", on_bad_lines='skip')

    print(len(df))
    print(df.head(5))
    embeddings = df['embeddings'].tolist()
    data_label = df['label'].tolist()

    embedd_vectors = [ast.literal_eval(embed_vec) for embed_vec in embeddings]
    cluster_pred, tsne_vectors = analyzer.get_clusters(embedd_vectors, cluster_no=2, with_tsne=True)
    ax = analyzer.get_cluster_plot_with_tsne(tsne_vector=tsne_vectors, cluster_pred=data_label)
    plt.show()


def add_vector_to_df():
    #df = pd.read_csv("../data/2024/training/cleaned/tmp/all/plasma_training_dataset_0_1.csv", encoding="utf-8")
    df = pd.read_csv("../data/2024/training/cleaned/tmp/all/plasma_test_dataset_0_1.csv", encoding="utf-8")
    df_embed = analyzer.add_embedding_vector(df, bert_model_name='SCI_BERT')
    df_embed.to_csv("../data/2024/training/cleaned/tmp/all/plasma_test_dataset_0_1._SCIBERT_Embeddings.csv", encoding='utf-8',
                    index=False)
    print('Adding Embeddings task is finished!!')


def show_scatter(x, colors, labels):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 2))

    # We create a scatter plot.
    f = plt.figure(figsize=(20, 20))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=100,
                    c=palette[colors.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('on')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(2):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(labels[i]), fontsize=16)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def show_docs_in_space():
    df = pd.read_csv("../data/training/cleaned/final/plasma_traing_data_with_patBert_embeddings.csv", encoding="utf-8",
                     on_bad_lines='skip')

    embeddings = df['embeddings'].tolist()
    embedd_vectors = [ast.literal_eval(embed_vec) for embed_vec in embeddings]
    x_embeddings = compute_tsne(np.array(embedd_vectors))

    df.label = pd.factorize(df.label)[0]
    data_label = df['label'].tolist()
    print(data_label[:10])


    #embedd_vectors = [ast.literal_eval(embed_vec) for embed_vec in embeddings]
    cluster_labels = ['PLASMA-Realted Patents',
                      'No_PLASMA_related Patents']

    show_scatter(x_embeddings, np.array(data_label), cluster_labels)
    plt.savefig('tsne-clusters_with_labels.png', dpi=100)
    plt.show()





if __name__ == '__main__':
    analyzer = Analyzer()
    pre = PatentTextPreProcessor()
    txt = ' This invention relates to a low-temperature dry etching method. More particularly, it relates to a low-temperature dry etching method which is suited for high-precision dry etching and in which etching is carried out by controlling the surface temperature '
    #print(pre.text_preprocessing_pipeline(txt))

    #txt = ' This invention relates to a low-temperature dry etching methods. More particularly, it relates to a low-temperature dry etching method which is suited for high-precision dry etching and in which etching is carried out by controlling the surface temperature '
    #print(analyzer.get_concepts(input_text=txt, algorithm='spacy'))


    #cv = analyzer.get_topic_coherence_plot(ab, min_k=15, max_k=40)
    #print(cv)
    # datales

    #concept extrcation
    #txt = "FIELD OF THE INVENTION \n [DESC0002] The present invention relates to a plasma processing apparatus and a processing gas supply structure thereof."
    #print(analyzer.get_concepts(input_text=txt, algorithm='spacy', normalized=False))
    #print(analyzer.get_concepts(input_text=txt, algorithm='textacy', normalized=True))

    #print(annotate_plasma_patent(txt))
    #add_vector_to_df()

    #show Tsne
    #show_tsne()
    #show_docs_in_space()

    #add_vector_to_df()
    show_tsne()












