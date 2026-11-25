#!/usr/bin/env python3
"""Compile tcn.onnx into a TensorRT engine ON THE EDGE DEVICE (engines are specific to the GPU / TensorRT version).

Tuned for FAST COMPILATION rather than fastest inference:
  * builder optimisation level 0 (least tactic search; TensorRT >= 8.6)
  * fixed batch shape (min = opt = max), so a single optimisation profile with one shape to tune
  * small workspace, FP32 by default (FP16 adds tactics to try; enable with --fp16 if you want lower latency)
  * on-disk timing cache: rebuilding the same / similar model afterwards reuses earlier tactic timings

Only needs the `tensorrt` python package (JetPack ships it); PyTorch is not required for compiling.

    python compile_trt.py                       # tcn.onnx -> tcn.engine
    python compile_trt.py --fp16 --batch 8
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")   # faster CUDA start-up

import tensorrt as trt

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", default=os.path.join(HERE, "tcn.onnx"))
    p.add_argument("--engine", default=os.path.join(HERE, "tcn.engine"))
    p.add_argument("--batch", type=int, default=1, help="fixed batch size baked into the engine (default 1)")
    p.add_argument("--fp16", action="store_true", help="allow FP16 kernels (faster inference, slower build)")
    p.add_argument("--opt-level", type=int, default=0, choices=range(0, 6),
                   help="builder optimisation level 0-5; 0 = fastest build (default), 3 = TensorRT default")
    p.add_argument("--workspace-mb", type=int, default=512, help="max scratch memory for tactics (default 512 MB)")
    p.add_argument("--timing-cache", default=os.path.join(HERE, "tcn.timing_cache"),
                   help="timing cache file, created if missing; pass '' to disable")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def build_engine(args):
    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TensorRT 10 networks are always explicit-batch (the flag is deprecated there); TensorRT 8 needs it.
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(args.onnx):
        for i in range(parser.num_errors):
            print("ONNX parse error:", parser.get_error(i), file=sys.stderr)
        sys.exit("Failed to parse " + args.onnx)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = args.opt_level
    else:
        print("WARNING: this TensorRT has no builder_optimization_level (needs >= 8.6); using its default effort")

    if args.fp16:
        if not hasattr(trt.BuilderFlag, "FP16"):                    # removed in TensorRT 11 (strongly typed networks only)
            print("WARNING: this TensorRT has no BuilderFlag.FP16 (TensorRT 11); building FP32")
        elif getattr(builder, "platform_has_fast_fp16", True):
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("WARNING: this GPU has no fast FP16; building FP32")

    # Pin the (dynamic) batch dimension of the ONNX model to a single shape.
    inp = network.get_input(0)
    if any(d < 0 for d in inp.shape):
        shape = (args.batch,) + tuple(inp.shape)[1:]
        profile = builder.create_optimization_profile()
        profile.set_shape(inp.name, shape, shape, shape)
        config.add_optimization_profile(profile)
    print(f"TensorRT {trt.__version__} | input '{inp.name}' {tuple(inp.shape)} -> batch {args.batch} "
          f"| opt level {args.opt_level} | {'FP16' if args.fp16 else 'FP32'}")

    cache = None
    if args.timing_cache:
        blob = open(args.timing_cache, "rb").read() if os.path.exists(args.timing_cache) else b""
        cache = config.create_timing_cache(blob)
        if not config.set_timing_cache(cache, False):      # cache from another GPU / TensorRT: start empty
            cache = config.create_timing_cache(b"")
            config.set_timing_cache(cache, False)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        sys.exit("Engine build failed (re-run with --verbose)")

    with open(args.engine, "wb") as f:
        f.write(serialized)
    if cache is not None:
        with open(args.timing_cache, "wb") as f:
            f.write(memoryview(cache.serialize()))
    print(f"Built {args.engine} ({memoryview(serialized).nbytes / 1024:.0f} KB) in {build_s:.1f} s")


if __name__ == "__main__":
    build_engine(parse_args())
