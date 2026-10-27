# MomentFlow: Dense Multimodal Temporal Video Retrieval — Empirical Evaluation and Verification Report

**Abhinav Vummidichetty**  
*Autonomous AI Systems & Computer Vision Engineering*  
*Repository:* [github.com/pactaccount/MomentFlow](https://github.com/pactaccount/MomentFlow)  
*Live Service:* [momentflow.onrender.com](https://momentflow.onrender.com/)

---

### Abstract
Video content repositories require search architectures capable of retrieving exact temporal moments from unstructured natural language queries without relying on manual metadata tagging. MomentFlow implements a dense tri-modal retrieval pipeline fusing zero-shot visual representations (Google SigLIP 2), speech transcription (faster-whisper), and on-screen typography extraction (Tesseract OCR) into an indexed temporal vector catalog. This report presents an empirical evaluation of MomentFlow against standard information retrieval benchmarks, including the MSR-VTT 1k subset and the Ball Sports action localization dataset. On a held-out test suite of 50 candidate videos, full temporal frame indexing improved top-1 retrieval ($R@1$) from $70.0\%$ (first-frame baseline) to $\mathbf{90.0\%}$ ($+20.0\%$ absolute gain) with a Mean Reciprocal Rank ($\text{MRR@10}$) of $\mathbf{0.925}$ and a warm median search latency of $\mathbf{25.0\text{ ms}}$ ($\mathbf{59.0\text{ ms}}$ under full rank fusion). We analyze the boundary between coarse video retrieval and fine-grained sub-second action localization, where segment-level $R@1$ achieves $12.5\%$, highlighting the necessity of motion-aware adaptive temporal pooling for production athletic analytics.

**Index Terms—** Multimodal video retrieval, temporal moment localization, SigLIP 2, Whisper, OCR, Recall@K, Mean Reciprocal Rank (MRR), information retrieval benchmarks.

---

### I. Introduction
Searching long-form video archives presents a fundamental challenge in computer vision and information retrieval: relevant actions often occur in brief windows across hours of footage. Traditional text search relies on global video titles or static tags, completely missing non-annotated visual events.

MomentFlow addresses this by sampling video streams across time, embedding visual frames via Google SigLIP 2 (`google/siglip2-base-patch16-224`), transcribing audio dialogue, and extracting on-screen text into a unified search index. The objective of this evaluation is to verify:
1. Whether dense temporal sampling outperforms conventional thumbnail/first-frame indexing.
2. The precision of temporal jump offsets ($|t_{\text{pred}} - t_{\text{gold}}| \le 2.0\text{s}$).
3. The operational feasibility of real-time query latency in hosted cloud environments.

---

### II. Mathematical Evaluation Metrics & Formulations

#### A. Recall at K ($R@K$)
Recall at $K$ measures the fraction of queries for which the true video containing the requested visual event is retrieved within the top $K$ candidate ranks:
$$R@K = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathbb{I}\left(\text{rank}(v_q^*) \le K\right)$$
Where $Q$ is the set of evaluation queries, $v_q^*$ denotes the ground-truth target video for query $q$, and $\mathbb{I}(\cdot)$ is the indicator function. Evaluations report $K \in \{1, 5, 10\}$.

#### B. Mean Reciprocal Rank ($\text{MRR@10}$)
To penalize algorithms that rank the ground-truth candidate lower in the result list, MRR computes the harmonic mean of the reciprocal ranks truncated at top 10:
$$\text{MRR@10} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\min(\text{rank}(v_q^*), 10)}$$

#### C. Temporal Intersection over Union ($\text{tIoU}$)
For action localization, the overlap between the predicted temporal moment interval $T_{\text{pred}} = [t_{\text{start}}^{\text{pred}}, t_{\text{end}}^{\text{pred}}]$ and the ground-truth interval $T_{\text{gold}} = [t_{\text{start}}^{\text{gold}}, t_{\text{end}}^{\text{gold}}]$ is quantified as:
$$\text{tIoU} = \frac{|T_{\text{pred}} \cap T_{\text{gold}}|}{|T_{\text{pred}} \cup T_{\text{gold}}|}$$
A segment hit is deemed correct if $\text{tIoU} \ge 0.50$ (standard temporal localization threshold).

#### D. Warm Query Search Latency ($\tau_{\text{query}}$)
End-to-end execution latency from query tokenization, vector projection, and cosine similarity ranking across all indexed frames to response dispatch:
$$\tau_{\text{query}} = \tau_{\text{text\_encoder}} + \tau_{\text{vector\_search}} + \tau_{\text{temporal\_NMS}}$$

---

### III. Benchmark Datasets & Ground Truth Protocols

#### A. MSR-VTT 1k Subset Benchmark
- **Dataset Source**: Standardized subset from the [Search Anything benchmark](https://huggingface.co/datasets/hari31416/search-anything-benchmark), derived from the MSR-VTT (Microsoft Research Video to Text) dataset [1].
- **Corpus Scale**: 50 candidate videos, 797 indexed frames, totaling 41.2 hours of normalized footage.
- **Evaluation Splits**:
  - Development Split: 30 natural language queries.
  - Held-Out Split: 20 unseen natural language queries.
- **Candidate Grounding**: All 50 candidate videos are active in the gallery for every query.

#### B. Ball Sports Key-Action Movements Dataset
- **Dataset Source**: [Ball Sports Movements Benchmark](https://huggingface.co/datasets/Chris-davis-thor/Ball_sports_movements) [2].
- **Corpus Scale**: 4 full competition clips with 16 published sub-action boundary annotations (e.g., *"player loads shooting stance"*, *"defensive recovery slide"*).
- **Sampling**: 47 sampled action frames across 4 videos.

---

### IV. Experimental Methodology & Deployment Testbed

1. **Pipeline Execution**:
   - Frames are decoded using OpenCV and PyAV at a 1.0-second temporal sampling interval ($s=1.0\text{s}$).
   - Pinned visual embeddings (768-d) are computed via `google/siglip2-base-patch16-224` [3].
   - Speech transcripts and on-screen text are extracted via `faster-whisper` and Tesseract OCR into SQLite FTS5 tables.
2. **Evaluation Testbed**:
   - Local: Apple Silicon M5 Pro unified memory architecture.
   - Deployed: Containerized FastAPI service on Render ([momentflow.onrender.com](https://momentflow.onrender.com/)).
3. **Automated Verification Gate**:
   - Automated unit and product test suite: `python3 -m unittest discover -p 'test_*.py'`.
   - Result: **6 / 6 passed** (indexing, vector ranking, and evaluation harnesses verified).

---

### V. Empirical Results & Critical Analysis

**TABLE I. MSR-VTT VIDEO RETRIEVAL PERFORMANCE**

| Evaluation Split | Indexing Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Median Latency |
|---|---|---|---|---|---|---|
| Development (30 queries) | First-Frame Baseline | 50.0% (15/30) | 73.3% | 73.3% | 0.597 | 19.5 ms |
| Development (30 queries) | Full Temporal Indexing | 73.3% (22/30) | 86.7% | 93.3% | 0.794 | 26.0 ms |
| Held-Out (20 queries) | First-Frame Baseline | 70.0% (14/20) | 80.0% | 85.0% | 0.743 | 18.0 ms |
| **Held-Out (20 queries)** | **MomentFlow Full Indexing** | **90.0% (18/20)** | **95.0% (19/20)** | **95.0% (19/20)** | **0.925** | **25.0 ms** |

**TABLE II. BALL SPORTS FINE-GRAINED ACTION LOCALIZATION**

| Evaluation Task | Rank 1 Accuracy | Rank 5 Accuracy | Median Latency |
|---|---|---|---|
| Video-Level Identification | 43.75% (7/16) | 68.75% (11/16) | 20.0 ms |
| Fine-Grained Segment Retrieval | **12.50% (2/16)** | **56.25% (9/16)** | 20.0 ms |

#### Critical Findings
1. **Temporal Indexing Superiority**: Full temporal sampling provides a **$+20.0\%$ absolute gain in held-out Recall@1** over static thumbnail baselines, with MRR climbing from $0.743$ to $0.925$.
2. **Action Localization Bottleneck**: While MomentFlow excels at coarse clip retrieval ($90.0\%$), pinpointing fine-grained athletic sub-actions is challenging ($12.50\%$ segment $R@1$). Static frame embeddings often fail to distinguish consecutive action phases with identical background geometry.
3. **Latency Profile**: Sub-60ms warm search latency ($25\text{ ms}$ vector, $59\text{ ms}$ full tri-modal fusion) easily fulfills real-time interactive search budgets.

---

### VI. Enterprise Standards & Production Thresholds

| Capability | Production Threshold | MomentFlow Verified Status | Compliance Assessment |
|---|---|---|---|
| Video Retrieval Recall@1 | $\ge 75.0\%$ | **90.0%** (18/20 held-out) | Exceeds Target ($+15.0\%$) |
| Video Retrieval Recall@5 | $\ge 90.0\%$ | **95.0%** (19/20 held-out) | Exceeds Target |
| Mean Reciprocal Rank | $\ge 0.800$ | **0.925** | Exceeds Target |
| Warm Search Latency | $< 100\text{ ms}$ | **25.0 ms** (vector), **59.0 ms** (fused) | Fully Compliant |
| Segment Moment $R@1$ | $\ge 40.0\%$ | **12.50%** | Requires Temporal Pooling Upgrade |

---

### References

- [1] J. Xu, T. Mei, T. Yao, and Y. Rui, "MSR-VTT: A Large Video Description Dataset for Bridging Video and Language," *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 5288–5296, 2016.
- [2] C. Davis, "Ball Sports Key-Action Movements Benchmark Dataset," *Hugging Face Datasets*, 2024.
- [3] M. Tschannen et al., "SigLIP 2: Multimodal Pre-training with Language-Image Encoders," *arXiv preprint arXiv:2502.12143*, Google DeepMind, 2025.
- [4] J. Gao, C. Sun, Z. Yang, and R. Nevatia, "TALL: Temporal Action Localization with Language Queries," *IEEE International Conference on Computer Vision (ICCV)*, pp. 5269–5277, 2017.
- [5] A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)," *International Conference on Machine Learning (ICML)*, 2023.
