# Evaluation: zero-shot visual moment search

Date: 2026-10-21. Model: `google/siglip2-base-patch16-224`, revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. No fine-tuning, prompt optimization, paid API, or training data was used. These are **historical local CPU measurements** made before the selected Hugging Face Space transport and Render deployment. The checkpoint is the same, but latency and product quality must be measured again through the hosted workflow. The local model cache was removed during repository cleanup.

## Video retrieval: 50 public MSR-VTT clips

Source: [Search Anything benchmark](https://huggingface.co/datasets/hari31416/search-anything-benchmark), a fixed 50-video subset of MSR-VTT test_1k with 30 development and 20 held-out text queries. It is **not** the official full MSR-VTT test_1k benchmark. All 50 videos are candidates for every query. Only media files are indexed; text queries and relevance labels are read after indexing. Video score is the highest-scoring sampled frame, with one rank per video. The first-frame baseline uses the same model and index, restricted to the first frame of each video.

| Split | Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Development, 30 queries | First frame only | 50.0% | 73.3% | 73.3% | 0.597 |
| Development, 30 queries | One frame/second | 73.3% | 86.7% | 93.3% | 0.794 |
| Held out, 20 queries | First frame only | 70.0% (14/20) | 80.0% | 85.0% | 0.743 |
| Held out, 20 queries | One frame/second | **90.0% (18/20)** | **95.0% (19/20)** | **95.0%** | **0.925** |

The sampling change gained **20 percentage points in held-out Recall@1** on this fixed subset. The product indexed 797 frames from 50 clips in about 34 seconds on the first run, including model loading. Warm median query time was about 25 ms. These timings are local measurements and may vary by hardware and cache state. The selected subset is small and likely easier than general video search; do not present 90% as a universal model accuracy or an official leaderboard result. The [dataset card](https://huggingface.co/datasets/hari31416/search-anything-benchmark) says the original MSR-VTT video rights remain with their owners, so media files are downloaded locally and excluded from Git.

### Product ranker compatibility check

The 50-video visual index was copied to an isolated test database and migrated through `ProductService`. The **actual multimodal product search method**, with no speech or OCR rows in this visual-only dataset, returned the same held-out video-level Recall@1 **90.0% (18/20)** and Recall@5 **95.0% (19/20)**. Development results remained 73.3% and 86.7%. Warm median search latency was 59 ms on this Mac. This checks that the rank fusion and five-second grouping preserved the earlier visual retrieval result on the fixed corpus; it does not measure the new speech/OCR indexing path or moment localization quality. The temporary compatibility-test index was removed during cleanup.

The original query-level ranks are preserved in [`evaluation_results/msrvtt_results.json`](evaluation_results/msrvtt_results.json). Re-run `python benchmark_msrvtt.py` against the same dataset after configuring the Hugging Face Space to compare the hosted path. The script checks SHA-256 for every clip.

## Action segment retrieval: four public sports clips

Source: [Ball Sports Key-Action Video Dataset](https://huggingface.co/datasets/Chris-davis-thor/Ball_sports_movements), four clips with 16 published start/end action annotations. The exact published segment descriptions are used as queries; all four videos are indexed. A hit counts as a matching moment only when it names the correct clip and its sampled timestamp lies inside the published action interval.

| Metric | Result |
| --- | ---: |
| Correct video at rank 1 | 43.75% (7/16) |
| Correct video by rank 5 | 68.75% (11/16) |
| Correct action segment at rank 1 | **12.5% (2/16)** |
| Correct action segment by rank 5 | 56.25% (9/16) |

The four clips contributed 47 sampled frames. First indexing took about 10.2 seconds total, including model loading; warm median query time was 20 ms. This dataset is too small for a reliable population estimate, but it exposes the main product weakness: one-frame-per-second image embeddings often cannot distinguish actions with similar static scenes, such as a soccer shot versus a save or different phases of a tennis rally. The current app should be described as **visual moment search with approximate timestamps**, not robust fine-grained temporal action recognition. Query-level ranks are preserved in [`evaluation_results/ball_sports_results.json`](evaluation_results/ball_sports_results.json).

## Earlier prototype checks

- Fake-encoder unit test passed for indexing, ranking, and evaluation logic.
- Real SigLIP 2 embeddings were finite and correctly ranked a red image over a blue image for a red-image query.
- FastAPI upload → background indexing → search passed on a synthetic short video.
- An earlier Docker image with local CPU PyTorch served `/api/health`, accepted a real public clip, indexed it, and returned timestamped search hits. The browser UI displayed those hits and playable videos.
- An earlier three-container Compose stack passed upload validation, duplicate detection, byte-range video playback, origin rejection, and persistence after worker restart. These do **not** validate the current Render Free topology; its sign-in behavior was removed for this public demo.
- Two six-second synthetic clips verified the **actual** Tesseract and faster-whisper paths: screen text `Billing Settings` and spoken words `quarterly sales report` both ranked the labeled clip and timestamp first. These are functional smoke checks, **not** a retrieval-accuracy benchmark.
- `evaluate.py` now calls `ProductService` and reports overall and per-modality timestamp recall from human-labeled manifests. It can be run against the same container image used for deployment.

## Next measurements

For a stronger portfolio claim, evaluate at least 100 held-out queries over a larger fixed candidate pool and add human-checked temporal boundaries from diverse clips. Include failure categories (motion-only, small objects, fast cuts, ambiguous captions), indexing throughput, RAM, cold start time, and concurrency. Keep all future comparisons on the same media, labels, hardware, and ranking rules. This work does not require model fine-tuning.
