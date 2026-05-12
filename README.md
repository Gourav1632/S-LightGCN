# S-LightGCN

## Semantic-Aware LightGCN for Cold-Start Recommendation Systems

S-LightGCN is a hybrid recommendation framework that enhances graph-based collaborative filtering by integrating semantic information from textual reviews into the initialization stage of LightGCN.

The project focuses on improving recommendation quality in sparse interaction and cold-start scenarios while preserving the simplicity and efficiency of LightGCN.

---

# Overview

Traditional collaborative filtering models struggle when users or items have very few interactions.
LightGCN improves recommendation by propagating embeddings across a user-item interaction graph, but it still relies heavily on graph topology.

S-LightGCN addresses this limitation by incorporating semantic embeddings extracted from textual reviews using Sentence-BERT before graph propagation begins.

Instead of random initialization:

* Semantic priors provide meaningful starting embeddings
* Collaborative signals propagate more effectively
* Cold-start recommendation quality improves

---

# Key Features

* Semantic-aware initialization for item embeddings
* LightGCN-based graph collaborative filtering
* Sentence-BERT textual feature extraction
* Cold-start recommendation enhancement
* BPR optimization objective
* Evaluation using Recall@K and NDCG@K
* Amazon Movies & TV dataset support

---

# Model Architecture

```text
Text Reviews
      ↓
Sentence-BERT Embeddings
      ↓
Linear Projection Layer
      ↓
Semantic Item Initialization
      ↓
LightGCN Graph Propagation
      ↓
Recommendation Prediction
```

---

# Tech Stack

* Python
* PyTorch
* LightGCN
* Sentence-Transformers
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

# Dataset

The experiments are conducted on the Amazon Movies and TV dataset.

Processed dataset statistics:

* 113,819 users
* 3,886 items
* 251,485 interactions

Only positive interactions (ratings >= 4) are treated as implicit feedback.

---

# Evaluation Metrics

The model is evaluated using:

* Recall@10
* Recall@20
* NDCG@10
* NDCG@20
* Cold-start Recall metrics

---

# Results

| Model      | Recall@10  | Recall@20  | NDCG@10    | NDCG@20    |
| ---------- | ---------- | ---------- | ---------- | ---------- |
| LightGCN   | 0.0866     | 0.1239     | 0.0511     | 0.0605     |
| S-LightGCN | **0.0950** | **0.1374** | **0.0530** | **0.0636** |

S-LightGCN improves both retrieval and ranking quality while maintaining the lightweight nature of LightGCN.

# Paper

**Semantic-Aware LightGCN: Enhancing Graph Collaborative Filtering with Textual Semantic Priors**

