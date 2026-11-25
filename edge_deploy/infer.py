#!/usr/bin/env python3
"""Run the regular PyTorch TCN and the compiled TensorRT engine on the same windows and compare them.

Needs: torch (with CUDA, as shipped for Jetson) and tensorrt. Reports, per backend:
  * accuracy / macro-F1 on the bundled test windows (sample_windows.npz)
  * single-batch latency (median / p95 / mean, input already on the GPU) and throughput
and, between the two: prediction agreement and the largest logit difference.

    python infer.py                         # PyTorch (GPU) vs TensorRT
    python infer.py --cpu --runs 500        # also time PyTorch on the CPU
"""
import argparse
import json
import os
import time

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import numpy as np
import torch

from model_def import CLASSES, N_CHANNELS, WINDOW, load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))


class TRTRunner:
    """Callable wrapper around a serialized engine; uses torch CUDA tensors as I/O buffers (no pycuda needed)."""

    def __init__(self, engine_path):
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"Could not load {engine_path} - was it built on this device with this TensorRT version?")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.in_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)

        shape = tuple(self.engine.get_tensor_shape(self.in_name))
        if min(shape) < 0:                                  # profile-based engine: use its (fixed) batch
            shape = tuple(self.engine.get_tensor_profile_shape(self.in_name, 0)[2])
            self.ctx.set_input_shape(self.in_name, shape)
        self.batch = shape[0]
        self.inp = torch.zeros(shape, dtype=torch.float32, device="cuda")
        self.out = torch.zeros(tuple(self.ctx.get_tensor_shape(self.out_name)), dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address(self.in_name, self.inp.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())

    def __call__(self, x):
        self.inp.copy_(x)                                   # device-to-device, ordered on the current stream
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def latency(fn, x, device, warmup, runs):
    for _ in range(warmup):
        fn(x)
    sync(device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        sync(device)
        times.append(time.perf_counter() - t0)
    t = np.asarray(times) * 1e3
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)), "mean_ms": float(t.mean())}


@torch.no_grad()
def run_all(fn, X, batch, device):
    """Logits for every window, in fixed-size batches (last batch padded by repeating its final window)."""
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        n = len(xb)
        if n < batch:
            xb = torch.cat([xb, xb[-1:].expand(batch - n, -1, -1)])
        out.append(fn(xb)[:n].float().clone())
    sync(device)
    return torch.cat(out).cpu().numpy()


def macro_f1(y, p):
    f1s = []
    for c in range(len(CLASSES)):
        tp = np.sum((p == c) & (y == c))
        prec = tp / max(np.sum(p == c), 1)
        rec = tp / max(np.sum(y == c), 1)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.path.join(HERE, "tcn_weights.pt"))
    ap.add_argument("--engine", default=os.path.join(HERE, "tcn.engine"))
    ap.add_argument("--data", default=os.path.join(HERE, "sample_windows.npz"))
    ap.add_argument("--n-windows", type=int, default=0, help="use only the first N windows (default: all)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--cpu", action="store_true", help="also time the PyTorch model on the CPU")
    ap.add_argument("--json", default="", help="write the results to this JSON file")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch - TensorRT needs the GPU.")
    gpu = torch.device("cuda")
    trt_model = TRTRunner(args.engine)
    batch = trt_model.batch

    model, mu, sigma = load_checkpoint(args.weights, gpu)
    d = np.load(args.data)
    X_raw, y = d["X"], d["y"]
    if args.n_windows:
        X_raw, y = X_raw[:args.n_windows], y[:args.n_windows]
    X = torch.from_numpy(((X_raw - mu) / sigma).astype(np.float32)).to(gpu)     # same standardisation as training
    print(f"{len(X)} windows ({N_CHANNELS} x {WINDOW}) | engine batch {batch} | GPU: {torch.cuda.get_device_name(0)}")

    backends = {"PyTorch (GPU)": (model, gpu), "TensorRT": (trt_model, gpu)}
    if args.cpu:
        backends["PyTorch (CPU)"] = (load_checkpoint(args.weights, "cpu")[0], torch.device("cpu"))

    rows, logits = {}, {}
    for name, (fn, dev) in backends.items():
        Xd = X.to(dev)
        logits[name] = run_all(fn, Xd, batch, dev)
        pred = logits[name].argmax(1)
        lat = latency(fn, Xd[:batch], dev, args.warmup, args.runs)
        rows[name] = {"accuracy": float((pred == y).mean()), "macro_F1": macro_f1(y, pred), **lat,
                      "windows_per_s": 1e3 * batch / lat["median_ms"]}

    ref, cmp = logits["PyTorch (GPU)"], logits["TensorRT"]
    agree = float((ref.argmax(1) == cmp.argmax(1)).mean())
    max_diff = float(np.abs(ref - cmp).max())

    print(f"\n{'backend':15} {'acc':>7} {'macroF1':>8} {'median ms':>10} {'p95 ms':>8} {'mean ms':>8} {'windows/s':>10}")
    for name, r in rows.items():
        print(f"{name:15} {r['accuracy']:7.4f} {r['macro_F1']:8.4f} {r['median_ms']:10.3f} {r['p95_ms']:8.3f} "
              f"{r['mean_ms']:8.3f} {r['windows_per_s']:10.0f}")
    print(f"\nTensorRT vs PyTorch: {100 * agree:.2f}% identical predictions, max |logit diff| = {max_diff:.2e}")
    print(f"TensorRT speed-up over PyTorch (GPU): {rows['PyTorch (GPU)']['median_ms'] / rows['TensorRT']['median_ms']:.2f}x "
          f"(latency is per batch of {batch}, input already on the GPU)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"batch": batch, "agreement": agree, "max_logit_diff": max_diff, "backends": rows}, f, indent=2)


if __name__ == "__main__":
    main()
