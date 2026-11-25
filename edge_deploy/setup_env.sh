#!/usr/bin/env bash
# Create a virtual environment and install everything compile_trt.py / infer.py need, then verify it.
#
#   bash setup_env.sh                                  # venv in ~/trt-venv
#   VENV=/path/to/venv bash setup_env.sh
#   TORCH_INDEX=https://download.pytorch.org/whl/cu130 bash setup_env.sh      # Jetson: force a torch wheel index
#
# Why it looks the way it does (each point was a real failure):
#   * Ubuntu 24.04 refuses system-wide `pip install` (PEP 668)      -> everything goes into a venv, pip runs as `python -m pip`
#   * the PyPI `tensorrt` package is a wheel-stub that only builds on x86-64  -> on Jetson / aarch64 TensorRT comes from apt (NVIDIA's
#     Jetson repo) and the venv is created with --system-site-packages so it can import it
#   * PyPI's aarch64 torch is CPU-only (torch.version.cuda is None)  -> on Jetson a CUDA build is installed from a CUDA wheel index
#
# x86-64 + NVIDIA GPU (driver >= 525): numpy 1.26.4, torch 2.4.1 (CUDA 12.1 wheel), tensorrt 10.3.0 from PyPI / pypi.nvidia.com.
# Jetson / aarch64: TensorRT = whatever the NVIDIA apt repo offers (JetPack 7.2 / R39: 11.4.1), torch = latest CUDA wheel for the release.
set -euo pipefail

VENV="${VENV:-$HOME/trt-venv}"
# slow links (the torch CUDA wheels are large): generous timeouts and retries for every pip call
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

ARCH="$(uname -m)"
JETSON=0
if [ -f /etc/nv_tegra_release ] || [ "$ARCH" = "aarch64" ]; then JETSON=1; fi
L4T_MAJOR=""
if [ -f /etc/nv_tegra_release ]; then L4T_MAJOR="$(sed -n 's/^# R\([0-9]*\).*/\1/p' /etc/nv_tegra_release | head -n1)"; fi
say "Python $(python3 --version 2>&1 | cut -d' ' -f2) | $ARCH | $([ "$JETSON" = 1 ] && echo "Jetson/aarch64 (L4T R${L4T_MAJOR:-?})" || echo 'generic Linux x86-64')"

# 1. venv support
if ! python3 -c "import ensurepip, venv" 2>/dev/null; then
    say "Installing python3-full / python3-venv (sudo)"
    sudo apt-get update
    sudo apt-get install -y python3-full python3-venv python3-pip
fi

# 2. Jetson: TensorRT python bindings from the NVIDIA apt repo (must exist BEFORE the venv is checked)
if [ "$JETSON" = 1 ]; then
    if ! /usr/bin/python3 -c "import tensorrt" 2>/dev/null; then
        say "Installing TensorRT python bindings from apt (sudo)"
        sudo apt-get update
        if apt-cache policy python3-libnvinfer | grep -q "Candidate: (none)"; then
            die "apt has no python3-libnvinfer. Is the NVIDIA Jetson repo configured? (ls /etc/apt/sources.list.d/; apt-cache policy | grep -i nvidia)"
        fi
        sudo apt-get install -y python3-libnvinfer python3-libnvinfer-dev
    fi
    /usr/bin/python3 -c "import tensorrt; print('system tensorrt', tensorrt.__version__)" \
        || die "TensorRT installed but not importable by /usr/bin/python3"
fi

# 3. the venv (Jetson: must see the apt-installed tensorrt)
if [ ! -x "$VENV/bin/python" ]; then
    say "Creating virtual environment $VENV"
    if [ "$JETSON" = 1 ]; then python3 -m venv --system-site-packages "$VENV"; else python3 -m venv "$VENV"; fi
else
    say "Re-using virtual environment $VENV"
    if [ "$JETSON" = 1 ] && grep -q "include-system-site-packages = false" "$VENV/pyvenv.cfg"; then
        sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' "$VENV/pyvenv.cfg"
        echo "enabled system site-packages in $VENV/pyvenv.cfg"
    fi
fi
PY="$VENV/bin/python"
# (no `pip install --upgrade pip`: not needed, and it aborted the script on slow / flaky links)

# 4. packages
if [ "$JETSON" = 0 ]; then
    say "Installing numpy 1.26.4, torch 2.4.1, tensorrt 10.3.0 (x86-64)"
    "$PY" -m pip install "numpy==1.26.4" "torch==2.4.1"
    "$PY" -m pip install "tensorrt==10.3.0" --extra-index-url https://pypi.nvidia.com
else
    say "Installing numpy"
    "$PY" -m pip install numpy

    # CUDA wheel index for this L4T release (R38/R39 = JetPack 7 = CUDA 13). Override with TORCH_INDEX=...
    if [ -z "${TORCH_INDEX:-}" ]; then
        if [ -n "$L4T_MAJOR" ] && [ "$L4T_MAJOR" -ge 38 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu130"
        else
            TORCH_INDEX=""
        fi
    fi
    if "$PY" -c "import sys, torch; sys.exit(0 if torch.version.cuda else 1)" 2>/dev/null; then
        say "torch with CUDA already installed: $("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    else
        [ -n "$TORCH_INDEX" ] || die "No known CUDA torch wheel index for L4T R${L4T_MAJOR:-?}. Install NVIDIA's CUDA torch wheel for your JetPack into $VENV, or set TORCH_INDEX=<index url>, then re-run."
        if "$PY" -c "import torch" 2>/dev/null; then
            say "Removing the CPU-only torch ($("$PY" -c 'import torch; print(torch.__version__)'))"
            "$PY" -m pip uninstall -y torch
        fi
        say "Installing torch (CUDA) from $TORCH_INDEX"
        "$PY" -m pip install torch --index-url "$TORCH_INDEX"
    fi
fi

# 5. verify
say "Verifying"
"$PY" - <<'EOF'
import sys
import numpy, tensorrt, torch
print("numpy", numpy.__version__, "| torch", torch.__version__, "| torch CUDA", torch.version.cuda, "| tensorrt", tensorrt.__version__)
if torch.version.cuda is None:
    sys.exit("torch is a CPU-only build (torch.version.cuda is None); infer.py needs a CUDA build")
if not torch.cuda.is_available():
    sys.exit("torch cannot see the GPU: wrong CUDA wheel for this JetPack/driver, or the GPU is unsupported by this build (check `nvidia-smi` / `tegrastats`)")
x = torch.ones(8, device="cuda") * 2          # a real CUDA kernel, not just device discovery
assert float(x.sum().item()) == 16.0
print("GPU:", torch.cuda.get_device_name(0))
if int(tensorrt.__version__.split(".")[0]) >= 11:
    print("note: TensorRT 11 - the default FP32 build is expected to work; compile_trt.py --fp16 is untested on it")
EOF

cat <<EOF

Done. Activate the environment, then run the scripts:
    source $VENV/bin/activate
    python compile_trt.py
    python infer.py
EOF
