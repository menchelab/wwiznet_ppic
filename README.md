# wwiznet_mb  – wwiznet with metabolite example data

Static web build (graph viewer) plus a tiny Python static file server for local preview.

quickstart: `python server.py`


## Precomputed layouts

The example dataset ships with a batch of precomputed 3D layouts (`x_<name>`/`y_<name>`/`z_<name>` columns in `nodes.tsv`):

| base name(s) | family | what it does |
|---|---|---|
| `community_1` | Louvain community | Detects communities via `python-louvain` (emergent, not the a priori layer3 label — a community can span multiple layers), lays out the community meta-graph with ForceAtlas2, then locally arranges each community's members around its slot. |
| `community_shellz_1` | Community + shell blend | `community_1`'s real force-computed xy (unchanged) + z from `shell_1`'s radius-by-layer3 formula, applied per **community's predominant layer3** rather than per node — "communities sorted by predominant measurement layer". No Procrustes alignment needed since it reuses `shell_1`'s radius *formula*, not its coordinates. Logs how many communities are single-layer-pure vs. genuinely mixed (mixed ones get a lossy majority-vote z). |
| `domaintouch_1` | Centrality elevation | Reuses `fa2_1`'s xy, z = number of distinct domains (protein/transcript/bridge/metabolite) a node's own type_a **plus its 1-hop neighborhood** collectively touch (1-4) — catches low-degree nodes that quietly bridge multiple domains, a signal neither `type_a` nor centrality alone expresses. |
| `ensemble_1` | Procrustes blend | Generalized Procrustes alignment (Gower 1975) of `fa23d_1` + `mds_1` + `spectral_1` — three deliberately different objectives (local force clustering, global geodesic fidelity, algebraic connectivity) aligned via `scipy.linalg.orthogonal_procrustes` and weighted-averaged into one consensus layout. |
| `fa23d_1` | ForceAtlas2 | Native 3D ForceAtlas2 (no manual z-stack — all 3 axes come from the force simulation). |
| `fa23d_2` | ForceAtlas2 | Same as `fa23d_1` but cross-layer edges down-weighted, so layers separate organically in 3D. |
| `fa2_1` | ForceAtlas2 | Shared 2D force layout (xy) for the whole graph, with a fixed z offset per layer3 (protein / bridge / metabolite stacked as 3 planes). |
| `fa2_2` | ForceAtlas2 | Same as `fa2_1` but with `linlog=True, dissuade_hubs=True` — compresses hub-dominated clusters differently. |
| `fa2_3` | ForceAtlas2 | Same as `fa2_1` but cross-layer edges are down-weighted before layout, so layers spread apart in xy *before* the z-stack is applied. |
| `fa2pathway_1` | ForceAtlas2 | Real interaction edges get their attraction boosted by the Jaccard similarity of their endpoints' real `pathways_a` annotation (321/1689 edges currently get a nonzero boost) — real topology stays primary, pathway co-membership only reinforces it. Unannotated nodes/edges behave exactly like `fa2_1`. |
| `fa2rarity_1` | ForceAtlas2 | Full 10-bucket `edge_type_c` rarity weighting (-ln(count/total) per exact type, not the binary within/cross-layer split `fa2_3` uses) — the 28 rare metabolite-protein edges pull ~3.4x tighter than the 500 common metabolite-metabolite ones. |
| `fa2sep_1` | ForceAtlas2 | Each layer's induced subgraph (protein-only, bridge-only, metabolite-only edges) is laid out *independently*, then z-stacked. xy is left overlapping across layers (no artificial offset). |
| `fa2spectral_1` | ForceAtlas2 | Seeded with a spectral embedding instead of random init (the fCoSE trick from Cytoscape's Compound Spring Embedder) — usually better global structure / faster convergence than `fa2_1`. |
| `fr3d_1` | Fruchterman-Reingold | Native 3D spring layout (`nx.spring_layout`, unweighted). |
| `fr3d_2` | Fruchterman-Reingold | Same, with cross-layer edges down-weighted. |
| `hive3_1` | Hive plot | One radial spoke per layer3 (Krzywinski-style hive plot, adapted to 3 axes in 3D via `fibonacci_sphere(3)`), nodes ordered by degree along their axis (hubs near the shared origin). Pure deterministic geometry, no simulation. |
| `hive5_1` | Hive plot | 5-axis generalization of `hive3_1` — one spoke per raw `type_a` value (protein/transcript/protein,transcript/bridge/metabolite) instead of the coarsened 3-way `layer3`, so the 21 dual-typed nodes get their own visible spoke instead of being folded into "protein". |
| `hyp_1` | Hyperbolic radius | `r = 1 - tanh(degree/beta)` (inspired by "Hyperbolic Embedding of Multilayer Networks", arXiv:2505.20378) — degree-0 nodes land exactly on the boundary automatically, no fallback shell needed. Azimuth from `fa2_1`'s xy, polar angle banded by layer3. |
| `isomap_1` | Isomap | Kernel PCA on a k-NN-restricted geodesic distance matrix (giant component, `sklearn.manifold.Isomap`). A distinct embedding objective from both MDS (stress majorization on the full distance matrix) and spectral (Laplacian eigenmaps). |
| `kk3d_1` | Kamada-Kawai | Path-length-based 3D layout, computed on the giant component only (745/900 nodes); every other node placed via the deterministic fallback shell. Unweighted hop distance. |
| `kk3d_2` | Kamada-Kawai | Same, but cross-layer edges get 3x the hop-distance weight, explicitly pushing layers apart. |
| `landscape_1` | Feature-space embedding | "functional landscape" (Pfeil et al., *Nat. Commun.* 2021): per-node feature vector (edge_type_c composition fractions + type_a one-hot + Louvain community one-hot + log-degree) embedded via UMAP. Similarity is in **annotation space**, not graph-distance space — two nodes with similar profiles land close together even across different connected components. |
| `mds_1` | Classical MDS | Metric MDS on the giant component's shortest-path distance matrix. Unweighted hop count. |
| `mds_2` | Classical MDS | Same, with cross-layer hop length inflated (3x) before the distance matrix is built. |
| `metapath_1` | node2vec-style, typed | Custom random-walk generator biased 4x toward staying within the same `edge_type_c` domain (protein-protein, metabolite-metabolite, etc.) before crossing — node2vec's own p/q never look at edge type at all. Feeds directly into gensim's Word2Vec + UMAP. |
| `n2vbal_1` | node2vec + UMAP | node2vec random-walk embedding (p=1, q=1: balanced walks) on the giant component, reduced to 3D via UMAP (cosine metric). |
| `n2vbal_densmap_1` | node2vec + DensMAP | Same node2vec embedding as `n2vbal_1`, reduced via DensMAP (Narayan, Berger & Cho, *Nat. Biotechnol.* 2021) — already free in the installed `umap-learn` via `densmap=True`; preserves true local density instead of UMAP's default behavior of visually equalizing it everywhere. |
| `n2vbal_pacmap_1` | node2vec + PaCMAP | Same node2vec embedding as `n2vbal_1`, reduced via PaCMAP (Wang et al., JMLR 2021) instead of UMAP — the same technique the frontend already uses client-side for its own in-browser layout calc. Often more evenly spread than UMAP by design (balances near/mid-near/further point-pair sampling). Requires the `pacmap` package. |
| `n2vbal_phate_1` | node2vec + PHATE | Same node2vec embedding as `n2vbal_1`, reduced via PHATE (Moon et al., *Nat. Biotechnol.* 2019) instead of UMAP — PHATE is purpose-built for biological data, preserving both local *and* global/trajectory structure. Requires the `phate` package. |
| `n2vbal_tsne_1` | node2vec + openTSNE | Same node2vec embedding as `n2vbal_1`, reduced via classic t-SNE (van der Maaten & Hinton 2008) through `openTSNE` (actively maintained, much faster than sklearn's) instead of UMAP — tends to produce tighter, more visually separated clusters than UMAP/PaCMAP give on this data, at some cost to inter-cluster distance fidelity. Uses `negative_gradient_method='bh'` (Barnes-Hut) since openTSNE's FFT acceleration is 2D-only. Requires the `openTSNE` package. |
| `n2vbfs_1` | node2vec + UMAP | Same, p=0.25, q=4 — more BFS-like/homophily-biased walks (emphasizes local neighborhood structure). |
| `n2vbfs_pacmap_1` | node2vec + PaCMAP | Same as `n2vbal_pacmap_1` but p=0.25, q=4 (BFS-like walks). |
| `n2vdfs_1` | node2vec + UMAP | Same, p=4, q=0.25 — more DFS-like/structural-equivalence-biased walks (emphasizes role/bridging structure). |
| `n2vdfs_pacmap_1` | node2vec + PaCMAP | Same as `n2vbal_pacmap_1` but p=4, q=0.25 (DFS-like walks). |
| `n2vlayered_1` | node2vec + UMAP, layered | node2vec+UMAP run **separately** on the protein-only and metabolite-only same-layer subnetworks, each z-squeezed into a thin slab and offset apart (protein negative z, metabolite positive z). Bridge nodes are placed via Tutte/harmonic-style iterative relaxation — each bridge node's position converges to the average position of its graph neighbors, propagated from the two fixed layers, so a bridge node connecting only to protein neighbors lands near the protein slab, one connecting to both lands truly in between. A handful of bridge-only components with no path to either layer (6/152 currently) fall back to a Fibonacci-sphere shell around the combined centroid. |
| `paga_1` | Two-level coarse+fine | PAGA-style layout (Wolf et al., *Genome Biology* 2019): Louvain communities abstracted into a meta-graph weighted by observed-vs-expected cross-community connectivity (configuration-model null), laid out with ForceAtlas2, then every node initialized near its community's coarse position and refined with a full spring-layout pass over the real graph. Community structure drives *initialization* for real physics, unlike `community_1`'s fixed placement. |
| `pathway_landscape_1` | Feature-space embedding | The real-biology counterpart to `landscape_1`: TF-IDF-weighted `pathways_a` multi-hot → `TruncatedSVD` → UMAP(3D). `pathways_a` only directly annotates 428/900 nodes (zero metabolites — it's gene-centric), so unannotated nodes get an inferred profile via 1-2 hop neighbor mean-pooling (a metabolite infers a profile from the proteins it interacts with), lifting real coverage to 838/900 (93%); the remaining 62 with no annotated node within 2 hops fall back to the usual Fibonacci-sphere shell. |
| `raritytouch_1` | Centrality elevation | Reuses `fa2_1`'s xy, z = -ln(rarest `edge_type_c` count a node touches / total edges) — the opposite signal from `topodeg_1`/`topobet_1`: a low-degree node with one rare cross-domain edge outranks a hub whose edges are all common types. |
| `rwr_1` | Feature-space embedding | Same paradigm as `landscape_1` but the feature vector is a personalized-PageRank (random-walk-with-restart, r=0.9) visitation distribution per node — a diffusion-based feature space instead of a fixed annotation vector. |
| `shell_1` | Concentric shells | Pure deterministic geometry (no simulation): protein on an inner sphere, bridge on a middle shell, metabolite on an outer shell, radii scaled by node count. Node order within each shell groups same-component nodes together. |
| `shell_2` | Concentric shells | Same, with the radius order reversed (metabolite inner, protein outer). |
| `shell_rarity_1` | Concentric shells | New self-contained shell family (like `shell_1`/`shell_2`) but banded by the rarest `edge_type_c` a node touches, not `layer3` — rare cross-domain connectors form the innermost shell regardless of which layer they belong to. |
| `spectral_1` | Spectral embedding | Laplacian eigenmaps on the giant component's adjacency matrix (unweighted), with a small epsilon background added for numerical stability. |
| `spectral_2` | Spectral embedding | Same, with cross-layer adjacency entries down-weighted. |
| `topobet_1` | Centrality elevation | Reuses `fa2_1`'s xy, sets z from betweenness centrality. |
| `topodeg_1` | Centrality elevation | Reuses `fa2_1`'s xy, sets z from degree centrality ("importance elevation"). |


## Data

example dataset is in data/data/
* nodes.tsv
* edges.tsv
* config.json - visualisation presets/nodeset presets etc


### nodes.tsv
* id column (required)
* arbitrary number of extra columns

### edges.tsv
* source (required, same id as in nodes.tsv id)
* target (required, same id as in nodes.tsv id)

### special column types/names
* x_&lt;layout&gt;, y_&lt;layout&gt;. z_&lt;layout&gt; - precomputed layouts, layout dropdown will be populated with &lt;layout&gt;'s
* *_a - array columns, comma seperated list of strings (eg: annotations)
* *_kv - comma seperated key value pairs (eg: key1:xyz,keyx2:asfd, ---)
* *_n - force column to be interpreted as numeric, eg metrics like degree, etc
* *_c - force column to interpreted as categorical, eg clusternumber, classification string etc
* *_c_kv - categorical keyvalue
* hl3d - optional, for molecular 3d viewer highlighted sites, comma seperated numerical index



### config.json
example preset option, can be generated with your settings in the app
* change settings in Controls menu etc
* save a preset (in Presets folder chose a name and press Save Current Presets)
* export json file (menu right top, Controls -> Presets -> Export local storage)



## Requirements

- Python 3.7+ (stdlib only; no `pip install`)

## Run locally

```bash
python server.py
```

Open [http://127.0.0.1:8888/](http://127.0.0.1:8888/) in your browser. Change the `PORT` value in `server.py` if 8888 is already in use.

Press **Ctrl+C** in the terminal to stop the server.

## Your data



| Path | Role |
|------|------|
| `index.html` | Entry page |
| `assets/` | Bundled JavaScript (ES modules) |
| `data/` | Dataset files (e.g. TSV, `config.json`) |
| `server.py` | Serves this folder over HTTP with correct `.js` MIME types on Windows |

## Notes

- The server always serves the directory that contains `server.py`, regardless of your current working directory.

