"""
MomentFlow - Apple 2-Model Architecture Benchmark Suite
========================================================
Runs evaluation across:
  1. MSR-VTT (Model-Level Video-Text Retrieval - Apple MobileCLIP-B)
  2. Charades-STA (System-Level Action Grounding - Apple 2-Model Pipeline)
  3. YouCook2 (System-Level Procedural Recipe Grounding - Apple 2-Model Pipeline, Validation Split)
"""

import os
import sys
import json
import time
import urllib.request
import tarfile
import io
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("Prototype"))
from apple_encoder import AppleMobileClipEncoder, normalize
from apple_grounder import AppleFastVLMVerifier
from media_tools import sampled_frames

def compute_tiou(p_s, p_e, g_s, g_e):
    inter = max(0.0, min(p_e, g_e) - max(p_s, g_s))
    union = max(p_e, g_e) - min(p_s, g_s)
    return inter / union if union > 0 else 0.0

def run_benchmarks():
    print("=" * 70)
    print("MOMENTFLOW: APPLE 2-MODEL BENCHMARK SUITE")
    print("=" * 70)
    
    t0 = time.perf_counter()
    m1 = AppleMobileClipEncoder()
    m2 = AppleFastVLMVerifier()
    print(f"Models initialized in {time.perf_counter() - t0:.2f}s!\n")
    
    results = {}
    
    # -------------------------------------------------------------
    # 1. MSR-VTT Benchmark (Model 1: Apple MobileCLIP-B)
    # -------------------------------------------------------------
    msr_json = "/Users/abhinav/Desktop/models/data/msrvtt/msrvtt_test_1k.json"
    msr_vids_dir = "/Users/abhinav/Desktop/models/data/msrvtt/videos"
    if os.path.isfile(msr_json) and os.path.isdir(msr_vids_dir):
        print("Running MSR-VTT Test Set Evaluation (100 Clips)...")
        with open(msr_json) as f:
            msr_entries = json.load(f)
        msr_items = []
        for entry in msr_entries:
            vpath = os.path.join(msr_vids_dir, entry["video"])
            if os.path.isfile(vpath):
                msr_items.append({"video": entry["video"], "path": vpath, "caption": entry["caption"]})
            if len(msr_items) >= 100:
                break
        
        def get_video_vector(vpath, num_frames=8):
            cap = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
            indices = np.linspace(0, max(0, total - 1), num_frames, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, f = cap.read()
                if ret and f is not None:
                    frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            cap.release()
            if not frames:
                return np.zeros(512, dtype=np.float32)
            vecs = m1.images(frames)
            return normalize(np.mean(vecs, axis=0).reshape(1, -1))[0]
        
        t_v = time.perf_counter()
        msr_v_vecs = np.stack([get_video_vector(it["path"]) for it in msr_items])
        v_fps = (len(msr_items) * 8) / (time.perf_counter() - t_v)
        
        t_t = time.perf_counter()
        msr_t_vecs = np.stack([m1.text(it["caption"]) for it in msr_items])
        t_qps = len(msr_items) / (time.perf_counter() - t_t)
        
        sim_matrix = msr_t_vecs @ msr_v_vecs.T
        ranks = []
        for i in range(len(msr_items)):
            scores = sim_matrix[i]
            r = int(np.where(np.argsort(-scores) == i)[0][0]) + 1
            ranks.append(r)
        ranks = np.array(ranks)
        
        results["msrvtt"] = {
            "video_count": len(msr_items),
            "recall_1": float(np.mean(ranks <= 1) * 100.0),
            "recall_5": float(np.mean(ranks <= 5) * 100.0),
            "recall_10": float(np.mean(ranks <= 10) * 100.0),
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks)),
            "throughput_fps": v_fps,
            "query_qps": t_qps
        }
        print(f"MSR-VTT R@1: {results['msrvtt']['recall_1']:.1f}% | MedR: {results['msrvtt']['median_rank']:.1f}")

    # -------------------------------------------------------------
    # 2. YouCook2 Validation Benchmark (System Level)
    # -------------------------------------------------------------
    yc_dir = "eval_datasets/youcook2/validation"
    if os.path.isdir(yc_dir):
        print("\nRunning YouCook2 Official Validation Split Evaluation...")
        url = "http://youcook2.eecs.umich.edu/static/YouCookII/youcookii_annotations_trainval.tar.gz"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            tar = tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz")
            for m in tar.getmembers():
                if m.name.endswith(".json"):
                    yc_db = json.load(tar.extractfile(m))["database"]
                    break
        
        yc_files = sorted([f for f in os.listdir(yc_dir) if f.endswith(".mp4")])
        y_records = []
        for f in yc_files:
            vid = f.replace(".mp4", "")
            if vid not in yc_db: continue
            vpath = os.path.join(yc_dir, f)
            stamps, images = [], []
            for s, img in sampled_frames(vpath, interval=1.0, max_duration=600.0):
                stamps.append(s); images.append(img)
            if not images: continue
            f_vecs = m1.images(images)
            for a in yc_db[vid]["annotations"][:4]:
                g_s, g_e = float(a["segment"][0]), float(a["segment"][1])
                t_s = time.perf_counter()
                q_vec = m1.text(a["sentence"])
                best_i = int(np.argmax(f_vecs @ q_vec))
                pred_s, pred_e = max(0.0, stamps[best_i] - 5.0), stamps[best_i] + 5.0
                verified, exp = m2.verify_action(images[best_i], a["sentence"])
                y_records.append({"tiou": compute_tiou(pred_s, pred_e, g_s, g_e), "latency": time.perf_counter() - t_s})
        
        y_ious = np.array([r["tiou"] for r in y_records])
        results["youcook2_validation"] = {
            "video_count": len(yc_files),
            "step_count": len(y_records),
            "recall_03": float(np.mean(y_ious >= 0.3) * 100.0),
            "recall_05": float(np.mean(y_ious >= 0.5) * 100.0),
            "recall_07": float(np.mean(y_ious >= 0.7) * 100.0),
            "miou": float(np.mean(y_ious) * 100.0),
            "avg_latency": float(np.mean([r["latency"] for r in y_records]))
        }
        print(f"YouCook2 Val R@1 (0.3): {results['youcook2_validation']['recall_03']:.1f}% | mIoU: {results['youcook2_validation']['miou']:.1f}%")

    out_file = "Prototype/runtime/full_evaluation_report.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll benchmark results persisted to {out_file}!")

if __name__ == "__main__":
    run_benchmarks()
