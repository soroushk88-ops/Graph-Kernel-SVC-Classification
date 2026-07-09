# PROTEINS data

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.manifold import Isomap, SpectralEmbedding, LocallyLinearEmbedding

from grakel.kernels import ShortestPath, GraphletSampling, RandomWalk, WeisfeilerLehman
from grakel.datasets.base import read_data
import gdown
import warnings
warnings.filterwarnings('ignore')

PROTEINS = read_data("PROTEINS")
graph_data, y = np.array(PROTEINS.data), np.array(PROTEINS.target)
G_train, G_test, y_train, y_test = train_test_split(graph_data, y, test_size=0.1, random_state=42)

# Shortest Path kernel
g_kernel = ShortestPath(normalize = True)
K_train = g_kernel.fit_transform(G_train)
K_test = g_kernel.transform(G_test)
model = SVC()
model.fit(K_train, y_train)
y_pred = model.predict(K_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
print(f"PROTEINS dataset: Shortest Path kernel ===> acuuracy: {round(acc, 4)}, precision: {round(prec, 4)}, recall: {round(rec, 4)}")

manifolds_names = ['Isomap', 'LocallyLinearEmbedding', 'SpectralEmbedding']
manifolds = [Isomap, LocallyLinearEmbedding, SpectralEmbedding]

for j, manif in enumerate(manifolds):
    encoder = manif(n_components = 10)
    E_train = encoder.fit_transform(K_train)
    E_test = encoder.transform(K_test) if j < 2 else encoder.fit_transform(K_test)

    model = SVC()
    model.fit(E_train, y_train)
    y_pred = model.predict(E_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"PROTEINS dataset: Shortest Path kernel + {manifolds_names[j]} ===> acuuracy: {round(acc, 4)}, precision: {round(prec, 4)}, recall: {round(rec, 4)}")



# Graphlet Sampling kernel
g_kernel = GraphletSampling()
K_train = g_kernel.fit_transform(G_train)
K_test = g_kernel.transform(G_test)
model = SVC()
model.fit(K_train, y_train)
y_pred = model.predict(K_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
print(f"PROTEINS dataset: Graphlet Sampling kernel ===> acuuracy: {round(acc, 4)}, precision: {round(prec, 4)}, recall: {round(rec, 4)}")

manifolds_names = ['Isomap', 'LocallyLinearEmbedding', 'SpectralEmbedding']
manifolds = [Isomap, LocallyLinearEmbedding, SpectralEmbedding]

for j, manif in enumerate(manifolds):
    encoder = manif(n_components = 10)
    E_train = encoder.fit_transform(K_train)
    E_test = encoder.transform(K_test) if j < 2 else encoder.fit_transform(K_test)

    model = SVC()
    model.fit(E_train, y_train)
    y_pred = model.predict(E_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"PROTEINS dataset: Graphlet Sampling kernel + {manifolds_names[j]} ===> acuuracy: {round(acc, 4)}, precision: {round(prec, 4)}, recall: {round(rec, 4)}")



# Random Walk kernel
g_kernel = RandomWalk()
K_train = g_kernel.fit_transform(G_train)
K_test = g_kernel.transform(G_test)
model = SVC()
model.fit(K_train, y_train)
y_pred = model.predict(K_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
print(f"PROTEINS dataset: Random Walk kernel ===> acuuracy: {round(acc, 4)}, precision: {round(prec, 4)}, recall: {round(rec, 4)}")

manifolds_names = ['Isomap', 'LocallyLinearEmbedding', 'SpectralEmbedding']
manifolds = [Isomap, LocallyLinearEmbedding, SpectralEmbedding]

for j, manif in enumerate(manifolds):
    encoder = manif(n_components = 10)
    E_train = encoder.fit_transform(K_train)
    E_test = encoder.transform(K_test) if j < 2 else encoder.fit_transform(K_test)

    model = SVC()
    model.fit(E_train, y_train)
    y_pred = model.predict(E_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"PROTEINS dataset: Random Walk kernel + {manifolds_names[j]} ===> acuuracy: {round(acc, 4)}, precision: {round(prec, 4)}, recall: {round(rec, 4)}")



# Weisfeiler-Lehman kernel
g_kernel = WeisfeilerLehman(normalize = True)
K_train = g_kernel.fit_transform(G_train)
K_test = g_kernel.transform(G_test)
model = SVC()
model.fit(K_train, y_train)
y_pred = model.predict(K_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
print(f"PROTEINS dataset: Weisfeiler-Lehman kernel ===> acuuracy: {round(acc, 4)}, precision: {round(prec, 4)}, recall: {round(rec, 4)}")

manifolds_names = ['Isomap', 'LocallyLinearEmbedding', 'SpectralEmbedding']
manifolds = [Isomap, LocallyLinearEmbedding, SpectralEmbedding]

for j, manif in enumerate(manifolds):
    encoder = manif(n_components = 10)
    E_train = encoder.fit_transform(K_train)
    E_test = encoder.transform(K_test) if j < 2 else encoder.fit_transform(K_test)

    model = SVC()
    model.fit(E_train, y_train)
    y_pred = model.predict(E_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"PROTEINS dataset: Weisfeiler-Lehman kernel + {manifolds_names[j]} ===> acuuracy: {round(acc, 4)}, precision: {round(prec, 4)}, recall: {round(rec, 4)}")