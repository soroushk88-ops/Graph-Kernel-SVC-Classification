# Graph-Kernel-SVC-Classification

Transforming graph-structured data into tabular feature vectors using graph kernels, reducing dimensionality with manifold learning, and classifying with Support Vector Classification (SVC).

Graph data (molecules, social networks, biological structures) is hard to feed directly into standard ML classifiers. This project builds a full pipeline that converts graphs into a tabular representation a classifier can use, while comparing how different kernels and dimensionality-reduction techniques affect downstream accuracy.

## Pipeline

```
Graph dataset → Graph Kernel (→ feature vectors) → Manifold Learning (dimensionality reduction) → SVC → Evaluation
```

Graph kernels compared (convert graph structure into feature vectors):
- Shortest Path Kernel — compares shortest-path length distributions between graphs
- Graphlet Sampling Kernel — compares frequencies of small local subgraphs
- Random Walk Kernel — compares random-walk statistics between graphs
- Weisfeiler-Lehman Kernel — iteratively refines node labels to capture structural similarity

Manifold learning techniques compared (dimensionality reduction):
- Isomap
- Locally Linear Embedding (LLE)
- Spectral Embedding

Classifier: Support Vector Classification (SVC)

## Datasets

| Dataset | Task | Size |
|---|---|---|
| [MUTAG](https://cdb.ics.uci.edu) | Predict mutagenicity of nitroaromatic compounds on *S. typhimurium* | 188 graphs, binary classification |
| PROTEINS | Predict whether a protein structure is an enzyme | Binary classification |
| PTC_MR | Predict carcinogenicity of compounds in male rats | Binary classification |

All datasets represent molecules as graphs, with nodes as atoms and edges as chemical bonds. Evaluation uses 10-fold cross-validation with accuracy, precision, and recall.

## Results

MUTAG (best-performing dataset):

| Kernel | Manifold | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Random Walk | Isomap | 1.000 | 1.000 | 1.000 |
| Random Walk | LLE | 1.000 | 1.000 | 1.000 |
| Shortest Path | Isomap | 0.947 | 0.923 | 1.000 |
| Weisfeiler-Lehman | LLE | 0.947 | 1.000 | 0.917 |
| Graphlet Sampling | Isomap | 0.737 | 0.706 | 1.000 |

PROTEINS (best-performing configuration):

| Kernel | Manifold | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Graphlet Sampling | Isomap | 0.714 | 0.857 | 0.400 |
| Random Walk | Isomap | 0.657 | 0.714 | 0.333 |
| Weisfeiler-Lehman | Isomap | 0.629 | 0.750 | 0.200 |

Key findings:
- The Random Walk kernel combined with Isomap or LLE achieved perfect classification on MUTAG, showing that global random-walk structure strongly separates mutagenic from non-mutagenic compounds in this dataset.
- Spectral Embedding consistently underperformed relative to Isomap and LLE across all kernels and datasets.
- Performance on PROTEINS and PTC_MR was noticeably lower than on MUTAG, reflecting the harder, more heterogeneous structure of those datasets.

## Tech Stack

- Python
- [GraKeL](https://ysig.github.io/GraKeL/) — graph kernel computation
- scikit-learn — manifold learning (Isomap, LLE, Spectral Embedding) and SVC
- NumPy / pandas — data handling

## Presentation
A full slide deck walking through the methodology, kernel-by-kernel and manifold-by-manifold comparisons, and complete results is included in this repository.

Part of my graduate coursework in Computer Science (AI & Data Engineering) at Ca' Foscari University of Venice, exploring how structural information in graphs can be captured through kernels and made usable for standard classifiers.
