#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/cellreplay" >&2
  exit 2
fi

cellreplay_dir="$(realpath "$1")"
sudo apt-get update
sudo apt-get install -y \
  protobuf-compiler libprotobuf-dev autotools-dev dh-autoreconf iptables \
  pkg-config dnsmasq-base debhelper libxcb-present-dev libcairo2-dev \
  libpango1.0-dev libtar-dev python3

cd "$cellreplay_dir"
./autogen.sh
./configure
make -j"$(nproc)"
sudo make install

if [[ -f traces/cellular-traces.zip && ! -d traces/tmobile ]]; then
  python3 -m zipfile -e traces/cellular-traces.zip traces
fi

echo "CellReplay installed at $(command -v mm-cellular)"
