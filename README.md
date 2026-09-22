# scDEAN

**scDEAN: A Lightweight Adaptive Fusion Model for Clustering scRNA-seq Data**

Subhashis Chatterjee, Rohit Bose
Department of Mathematics and Computing, Indian Institute of Technology (ISM) Dhanbad

Official implementation of scDEAN, a deep clustering framework that integrates gene-expression
and cell–cell topological information through a lightweight softmax-gated dual-encoder fusion,
and formulates clustering as a differentiable centroid-optimisation task decoupled from
representation learning.

---

## Overview

scDEAN has three components:

| Module | What it does |
| --- | --- |
| **Graph Construction** | Builds a weighted cell–cell graph directly from the UMAP fuzzy simplicial set (correlation distance, `k = 15`), avoiding a PCA → kNN pipeline. |
| **Representation Learning** | A dual encoder: an FNN branch (`MLP`) modelling expression variability and a GCN branch modelling cell–cell topology. Their outputs are fused at *every* layer by a softmax gate with only `4d` parameters per layer. Trained with reconstruction, adjacency-BCE, graph-Laplacian and negative-binomial losses. |
| **Deep Clustering** | The encoder is **frozen**; only a small classifier and a set of learnable centroids are optimised, using fuzzy memberships plus a WCSS/BCSS distance-ratio regulariser. |

---

## Repository structure

```
scDEAN/
├── main.py                 # CLI entry point: full pipeline + evaluation
├── modules.py              # compatibility shim (re-exports everything; see note below)
├── requirements.txt
├── LICENSE
└── scdean/
    ├── __init__.py         # public API
    ├── data.py             # load_dataset, load_data, GCNDataset, collate_fn
    ├── layers.py           # GCNlayer, MLP, StudentskernalAdjacency
    ├── models.py           # GCNLatentRepresentation, DeepClustering
    ├── losses.py           # reconstruction / adjacency / graph-reg / NB losses
    ├── train.py            # train_gcn_with_batching, fine_tune_model, extract_latent_representations
    ├── metrics.py          # clustering_accuracy (Hungarian matching)
    └── utils.py            # set_seed, save/load_model_state
```

`modules.py` is kept as a thin re-export layer so that any script or notebook written
against the earlier single-file layout (`from modules import GCNDataset, ...`) still runs
unchanged.

---

## Installation

Tested with **Python 3.11** and **PyTorch 2.3.1** on Linux with NVIDIA GPUs.

```bash
git clone https://github.com/<user>/scDEAN.git
cd scDEAN

conda create -n scdean python=3.11 -y
conda activate scdean

# Install PyTorch matching your CUDA version first, e.g. for CUDA 12.1:
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric (must match the installed torch build)
pip install torch-geometric

pip install -r requirements.txt
```

> `torch-geometric` supplies `GCNConv`, used by `scdean/layers.py`. Install it *after*
> PyTorch and match the CUDA build, otherwise the GCN branch will fail to import.

---

## Data

Place each dataset under `--base_path` (default `./data`) in one of two layouts:

```
data/
├── Camp.h5                     # --data_type .h5   (expects f['X'] and f['obs/Group'])
└── Camp/                       # --data_type .csv
    ├── data.csv                #   cells x genes, first column = index
    └── label.csv               #   cell labels, first column = index
```

All datasets used in the paper are publicly available:

| Dataset | Cells | Clusters | Protocol | Source |
| --- | --- | --- | --- | --- |
| Yan | 90 | 7 | Tang | GEO `GSE36552` |
| Camp | 777 | 7 | SMARTer | GEO `GSE81252` |
| Petropoulos | 1,529 | 5 | Smart-seq2 | ArrayExpress `E-MTAB-3929` |
| Klein | 2,717 | 4 | inDrop | GEO `GSE65525` |
| Zeisel | 3,005 | 9 | STRT-Seq | GEO `GSE60361` |
| PBMC-Zheng | 4,340 | 8 | 10x Chromium | [10x Genomics pbmc4k](https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc4k) |
| Baron | 8,569 | 14 | inDrop | GEO `GSE84133` |
| AD-Brain | 13,214 | 8 | 10x, snRNA-seq | GEO `GSE138852` |

Biological validation additionally uses HNSCC (GEO `GSE103322`) and TCGA-HNSC bulk
RNA-seq + survival data from the [UCSC Xena GDC hub](https://xenabrowser.net/).

---

## Usage

Basic run:

```bash
python main.py --dataset_name Camp --data_type .h5 --base_path ./data
```

Batch size follows dataset size, as reported in the paper:

```bash
# < 4,000 cells  (Yan, Camp, Petropoulos, Klein, Zeisel)
python main.py --dataset_name Klein   --batch_size 256

# <= 8,000 cells (PBMC-Zheng)
python main.py --dataset_name PBMC-Zheng --batch_size 512

# > 8,000 cells  (Baron, AD-Brain)
python main.py --dataset_name Baron   --batch_size 1024
python main.py --dataset_name AD-Brain --batch_size 1024
```

Reproduce the initialisation-robustness analysis (Table 4) by sweeping the seed:

```bash
for s in 0 1 2 3 4; do
  python main.py --dataset_name Baron --batch_size 1024 --seed $s
done
```

Save the pretrained encoder so later runs skip Phase 1:

```bash
python main.py --dataset_name Camp --save_pt --model_dir ./checkpoints
```

### Key arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--n_top_genes` | `300` | Genes retained, ranked by standard deviation across cells |
| `--k_neighbors` | `15` | Neighbourhood size `k` for the UMAP fuzzy graph |
| `--hidden1 / --hidden2 / --latent_dim` | `128 / 64 / 32` | Encoder layer sizes (decoder mirrors these) |
| `--m` | `1.3` | Fuzzifier in the membership function |
| `--alpha / --beta / --gamma / --lambda_param` | `0.25` each | Weights for reconstruction / adjacency / graph-reg / NB losses |
| `--epochs` | `200` | Representation-learning epochs (Adam, lr `1e-3`) |
| `--fine_tune_epochs` | `100` | Deep-clustering epochs (Adam, lr `5e-3`, weight decay `1e-4`, StepLR ×0.5 every 30) |
| `--seed` | `42` | Global seed |
| `--recon_loss` | `nb` | Count-based regulariser: `nb` or `zinb`. Use `--lambda_param 0` for MSE only. |
| `--update_centroids` | off | Optional FCM-style centroid reassignment. **Off for all reported results.** |

The script prints ACC, NMI, ARI and the Silhouette Index, followed by total runtime and
peak GPU/CPU memory. Runtime and peak VRAM cover model training and clustering only,
excluding preprocessing.

---

## Reproducibility notes

`set_seed()` seeds Python, NumPy and PyTorch, enables `cudnn.deterministic`, sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and calls `torch.use_deterministic_algorithms(..., warn_only=True)`,
which is required because the scatter-add atomics inside `GCNConv` are otherwise
non-deterministic on CUDA.

Even so, exact bit-level reproduction across *different* machines is not guaranteed:
k-means initialisation, mini-batch composition (batch size varies with dataset size) and
GPU kernel scheduling can all shift results slightly. The magnitude of this variation is
quantified in Table 4 of the paper (e.g. ACC `0.762 ± 0.053` on Baron across five
k-means seeds), and is the reason the default configuration in each ablation table may
differ marginally from the main benchmark table.

---

## Implementation notes

A few places where the code is intentionally more specific than the equations in the paper:

- **Mean parameter** uses `softplus` rather than `exp`. The two give near-identical
  results, but `exp` was numerically unstable on some datasets; the dispersion uses
  `exp` as written.
- **Latent similarity** retains the diagonal: masking self-similarities was tested and
  reduced performance on Baron.
- **Loss reductions** are means rather than sums; the constant rescaling is absorbed into
  `alpha`/`beta`/`gamma`/`lambda_param`.
- **`DeepClustering.forward`** returns raw logits; the softmax is applied inside
  `CrossEntropyLoss`, so the optimised objective is unchanged.
- The `batch_norm_*` and `layer_norm_*` attributes on `GCNLatentRepresentation` are
  declared but not used in `forward`; they are retained for checkpoint compatibility.

---

## Citation

```bibtex
@article{chatterjee_scdean,
  title   = {scDEAN: A Lightweight Adaptive Fusion Model for Clustering scRNA-seq Data},
  author  = {Chatterjee, Subhashis and Bose, Rohit},
  journal = {Applied Intelligence},
  year    = {2026}
}
```

---

## License

Released under the MIT License - see [LICENSE](LICENSE).

## Contact

Rohit Bose — rohitbose269@gmail.com
Subhashis Chatterjee — subhashis@iitism.ac.in
