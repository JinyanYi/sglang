#!/usr/bin/env bash
# Build the checked-out SGLang router on an aarch64 Linux host.
set -Eeuo pipefail
trap 'rc=$?; echo "Router build failed at line $LINENO (exit $rc)" >&2' ERR

echo 'Preparing the SGLang router build'
[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || {
  echo 'Run this script on an aarch64 Linux host.' >&2
  exit 1
}
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
revision="$(git -C "$repo" rev-parse --verify "${1:-HEAD}^{commit}")"
build="$(mktemp -d /tmp/a5-router-build-XXXXXX)"
output_root=/mnt/share/y00970600/a5-router-wheels
mkdir -p "$output_root"
output="$(mktemp -d "$output_root/${revision:0:10}-XXXXXX")"
echo "source_commit=$revision"
echo "build_dir=$build"
echo "wheel_dir=$output"
printf '%s\n' "$revision" > "$output/source.commit"
getconf GNU_LIBC_VERSION | tee "$output/build-glibc.txt"

missing=()
for tool in python3 cc c++ make cmake perl curl git unzip pkg-config; do
  command -v "$tool" >/dev/null || missing+=("$tool")
done
if (( ${#missing[@]} )); then
  echo "Installing missing build tools: ${missing[*]}"
  [[ "$(id -u)" == 0 ]] || { echo 'Install build prerequisites as root first.' >&2; exit 1; }
  if command -v apt-get >/dev/null; then
    apt-get update
    apt-get install -y build-essential pkg-config cmake perl curl git unzip python3-venv
  elif command -v dnf >/dev/null; then
    dnf install -y gcc gcc-c++ make pkgconf-pkg-config cmake perl curl git unzip python3 python3-pip
  elif command -v yum >/dev/null; then
    yum install -y gcc gcc-c++ make pkgconfig cmake perl curl git unzip python3 python3-pip
  else
    echo 'Install the listed tools, then rerun this script.' >&2
    exit 1
  fi
fi

git -C "$repo" archive "$revision" sgl-model-gateway | tar -x -C "$build"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
if ! command -v rustup >/dev/null; then
  curl --proto '=https' --tlsv1.2 -fLsS https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain none --no-modify-path
fi
rustup toolchain install 1.90 --profile minimal --component clippy
curl -fL https://github.com/protocolbuffers/protobuf/releases/download/v32.0/protoc-32.0-linux-aarch_64.zip \
  -o "$build/protoc.zip"
unzip -q "$build/protoc.zip" -d "$build/protoc"
export PATH="$build/protoc/bin:$PATH"
protoc --version
if ! python3 -m venv "$build/venv"; then
  if [[ "$(id -u)" == 0 ]] && command -v apt-get >/dev/null; then
    apt-get update
    apt-get install -y python3-venv
    python3 -m venv "$build/venv"
  else
    echo 'Enable python3 venv/ensurepip support, then rerun this script.' >&2
    exit 1
  fi
fi
"$build/venv/bin/python" -m pip install 'maturin[patchelf]>=1,<2'
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
cd "$build/sgl-model-gateway/bindings/python"
"$build/venv/bin/maturin" build --release --features vendored-openssl \
  --interpreter "$build/venv/bin/python" --out "$output" 2>&1 | tee "$output/build.log"
(
  cd "$output"
  sha256sum ./*.whl > SHA256SUMS
  cat SHA256SUMS
)
echo "wheel_dir=$output"
