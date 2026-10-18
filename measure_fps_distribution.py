"""
measure_fps_distribution.py
Measuring the FPS distribution (mean, std, min, max) of DAEG-YOLO inference
on the ExDark test set: 10 warmup + 50 evaluation passes, batch=1, AMP, RTX 5080.
Report latency distributions, not only peak throughput."
Output: prints stats + saves to fps_distribution_result.json
"""
# ── APPLY THE PATCH BEFORE IMPORTING ANYTHING ────────────────────────────────────
import pickle as _pkl

class _SafeUnpickler(_pkl.Unpickler):
    _dummy_cache = {}
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError):
            key = f'{module}.{name}'
            if key not in self._dummy_cache:
                self._dummy_cache[key] = type(name, (), {
                    '__module__': module,
                    '__init__':     lambda s, *a, **kw: None,
                    '__setstate__': lambda s, st: None,
                })
            return self._dummy_cache[key]

_pkl.Unpickler = _SafeUnpickler
# ── END PATCH ──────────────────────────────────────────────────────────────

import sys, time, json, statistics
sys.stdout.reconfigure(encoding='utf-8')

from train_daeg_yolo import (
    DCENet9,
    _FPNCapture, _TrainState, _CondReconStats, _PatchedLoss,
    DAFMValidator, DAFMTrainer,
)  # noqa

import torch
from ultralytics import YOLO

import argparse as _ap
_parser = _ap.ArgumentParser()
_parser.add_argument("--weights", required=True, help="Path to best.pt")
_parser.add_argument("--data",    default="datasets/ExDark_Official/ExDark.yaml")
_parser.add_argument("--warmup",  type=int, default=10)
_parser.add_argument("--passes",  type=int, default=50)
_parser.add_argument("--out",     default="fps_distribution_result.json")
_args = _parser.parse_args()
WEIGHTS   = _args.weights
DATA_YAML = _args.data
WARMUP    = _args.warmup
EVAL_PASS = _args.passes
OUT_JSON  = _args.out

# ── All executable code must be contained within this guard ───────────────────────────
# Windows multiprocessing (spawn) reruns the script on each worker.
# Without this safeguard, the worker will attempt to run YOLO(WEIGHTS) again and crash.
if __name__ == '__main__':
    print("Loading model...")
    model = YOLO(WEIGHTS)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Warmup
    print(f"Running {WARMUP} warmup passes...")
    for _ in range(WARMUP):
        model.val(data=DATA_YAML, batch=1, device=device, half=True, split='test', verbose=False)

    # Measurement
    fps_list = []
    latency_ms_list = []
    print(f"Running {EVAL_PASS} evaluation passes...")
    for i in range(EVAL_PASS):
        results = model.val(data=DATA_YAML, batch=1, device=device, half=True, split='test', verbose=False)
        speed = results.speed
        total_ms = speed['preprocess'] + speed['inference'] + speed['postprocess']
        fps_list.append(1000.0 / total_ms)
        latency_ms_list.append(total_ms)
        if (i + 1) % 10 == 0:
            print(f"  Pass {i+1}/{EVAL_PASS}: {fps_list[-1]:.1f} FPS  ({total_ms:.2f} ms)")

    stats = {
        'fps_mean':          round(statistics.mean(fps_list), 2),
        'fps_std':           round(statistics.stdev(fps_list), 2),
        'fps_median':        round(statistics.median(fps_list), 2),
        'fps_min':           round(min(fps_list), 2),
        'fps_max':           round(max(fps_list), 2),
        'latency_mean_ms':   round(statistics.mean(latency_ms_list), 2),
        'latency_std_ms':    round(statistics.stdev(latency_ms_list), 2),
        'latency_median_ms': round(statistics.median(latency_ms_list), 2),
        'latency_min_ms':    round(min(latency_ms_list), 2),
        'latency_max_ms':    round(max(latency_ms_list), 2),
        'n_warmup': WARMUP,
        'n_eval':   EVAL_PASS,
        'device':   device,
        'batch':    1,
        'precision': 'AMP (FP16)',
    }

    print(f"\n=== FPS Distribution (n={EVAL_PASS}) ===")
    print(f"  Mean   : {stats['fps_mean']} FPS")
    print(f"  Std    : {stats['fps_std']} FPS")
    print(f"  Median : {stats['fps_median']} FPS")
    print(f"  Range  : {stats['fps_min']} -- {stats['fps_max']} FPS")
    print(f"\n=== Latency Distribution ===")
    print(f"  Mean   : {stats['latency_mean_ms']} ms")
    print(f"  Std    : {stats['latency_std_ms']} ms")
    print(f"  Range  : {stats['latency_min_ms']} -- {stats['latency_max_ms']} ms")

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")
