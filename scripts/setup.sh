#!/usr/bin/env bash
# Installs what scripts/tester_script.sh needs on Ubuntu or Debian (Colab
# included), then runs it: Rust if cargo is missing, and the CUDA compiler
# and NVRTC if an NVIDIA driver is there but nvcc isn't. Needs root or sudo.
set -euo pipefail
cd "$(dirname "$0")/.."

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO=sudo
fi

export PATH="$PATH:$HOME/.cargo/bin:/usr/local/cuda/bin"

if ! command -v cargo > /dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
fi

if command -v nvidia-smi > /dev/null && ! command -v nvcc > /dev/null; then
    # The newest CUDA the driver runs, like "12.4"; the toolkit can't be newer.
    cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p')
    # shellcheck disable=SC1091
    . /etc/os-release
    distro="${ID}${VERSION_ID//./}"
    keyring=$(mktemp --suffix=.deb)
    curl -sSfLo "$keyring" \
        "https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb"
    $SUDO dpkg -i "$keyring"
    rm -f "$keyring"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq "cuda-nvcc-${cuda/./-}" "cuda-nvrtc-dev-${cuda/./-}"

    # cudarc finds NVRTC through the loader.
    echo "/usr/local/cuda-$cuda/lib64" | $SUDO tee /etc/ld.so.conf.d/wilupgu-cuda.conf > /dev/null
    $SUDO ldconfig
    export PATH="/usr/local/cuda-$cuda/bin:$PATH"
fi

exec bash scripts/tester_script.sh
