# MomentFlow — résumé and demo evidence

## Problem and use case
Teams with long video libraries need to find a relevant clip and timestamp without manually watching every file. MomentFlow indexes sampled frames, speech, and visible text, then returns timestamped search hits. Possible uses include media review, training-video search, and visual incident triage. No customer time savings have been measured.

## Engineering approach
The FastAPI app accepts video uploads and queues indexing. A worker decodes frames, runs speech transcription and OCR, requests pinned SigLIP 2 visual embeddings from a Hugging Face Space, and stores searchable evidence in SQLite. Search fuses modalities and groups nearby hits into moments. The website presents a video library, results, and playback. Job recovery and encoder-contract tests cover some failure paths.

## Hard parts and limits
The hosted service depends on a separate private embedding Space. Render's free storage is ephemeral: uploaded videos can disappear after a restart. The visual search benchmark is stronger than the action-localization benchmark, so do not describe this as reliable fine-grained action detection. Public upload also needs deployment-specific abuse controls before a sustained public launch.

## Measured evidence
See [the evaluation](../Prototype/EVALUATION.md), [query-level results](../Prototype/evaluation_results/msrvtt_results.json), and [comparison chart](recall_chart.svg). On a fixed 50-video MSR-VTT subset, held-out video Recall@1 was 18/20 (90%) versus 14/20 (70%) for a first-frame baseline; Recall@5 was 19/20. The product search path reproduced 18/20 Recall@1 with 59 ms warm median query time on the local Mac. These are small-subset local measurements, not full MSR-VTT or hosted latency results. Action-segment Recall@1 was 2/16.

## Verification and proof
`cd Prototype && python3 -m unittest discover -p 'test_*.py'` passed 6 tests on 2026-10-22. The [hosted site](https://momentflow.onrender.com/) returned 200 after a roughly 34-second cold start. A small original nine-second sample video was uploaded; `/api/videos` reported **Ready**, `/api/health` reported nine indexed frames, and a search for “red car” returned the matching clip at 0 seconds first. Visitors can click **Load sample video** in an empty library, wait for indexing, then search “red car.” The deployed sample file and button were both verified available. This is a functional hosted smoke check, not hosted accuracy or latency evaluation. Link the GitHub evaluation with the hosted URL. Render's ephemeral storage means the sample may need re-upload after a restart; keep a short recording as stable backup proof.

## Résumé wording
“Built multimodal video search with timestamped playback; improved Recall@1 from 70% to 90% on a fixed 50-video, 20-query held-out subset, with 59 ms warm local search.” Keep the dataset size and “local” qualifier. Do not claim production adoption or general video-search accuracy.
