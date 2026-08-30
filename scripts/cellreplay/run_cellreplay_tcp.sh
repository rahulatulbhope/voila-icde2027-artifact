#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 TRACE_DIR MANIFEST OUTPUT_JSONL LOSS_RATE SERVER_HOST [PORT]" >&2
  exit 2
fi

trace_dir="$(realpath "$1")"
manifest="$(realpath "$2")"
output="$(realpath -m "$3")"
loss_rate="$4"
server_host="$5"
port="${6:-50127}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trace_name="$(basename "$(dirname "$trace_dir")")/$(basename "$trace_dir")"
packet_log_dir="$(dirname "$output")/packet_logs/${trace_name//\//_}_loss_${loss_rate}"
mkdir -p "$(dirname "$output")" "$packet_log_dir"

python3 "$script_dir/tcp_receiver.py" --port "$port" &
receiver_pid=$!
trap 'kill "$receiver_pid" 2>/dev/null || true' EXIT

mm-cellular 11 \
  "$trace_dir/up-delay-light-pdo" \
  "$trace_dir/down-delay-light-pdo" \
  "$trace_dir/up-heavy-pdo" \
  "$trace_dir/down-heavy-pdo" \
  --packet-log-folder="$packet_log_dir" \
  --loss-rate="$loss_rate" \
  --psize-latency-offset-up="$trace_dir/latency-offset-up" \
  --psize-latency-offset-down="$trace_dir/latency-offset-down" \
  --once \
  python3 "$script_dir/tcp_sender.py" \
  --host "$server_host" --port "$port" --manifest "$manifest" \
  --output "$output" --trace "$trace_name" --loss-rate "$loss_rate"

wait "$receiver_pid"
trap - EXIT
