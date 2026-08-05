#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  wait_then_run.sh --pid PID [--pid PID ...] [options] -- COMMAND [ARG ...]

Wait until all specified Linux processes have exited, then run COMMAND.

Options:
  --pid PID              Process to wait for. May be specified more than once.
  --interval SECONDS     Polling interval (default: 30).
  --settle-seconds SEC   Extra wait after process exit for resource cleanup
                         (default: 30).
  -h, --help             Show this help.

Example:
  nohup ./wait_then_run.sh --pid 12345 -- \
    env CUDA_VISIBLE_DEVICES=2,3,4 python3 pre-training/pretrain.py ... \
    > wait_and_pretrain.log 2>&1 &
EOF
}

log() {
    printf '[%(%F %T)T] %s\n' -1 "$*"
}

die() {
    log "ERROR: $*" >&2
    exit 2
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

read_process_identity() {
    local pid=$1
    local stat_line stat_rest
    local -a stat_fields

    [[ -r "/proc/${pid}/stat" ]] || return 1
    IFS= read -r stat_line < "/proc/${pid}/stat" || return 1

    # Everything after the final ") " starts at field 3 (process state).
    stat_rest=${stat_line##*) }
    read -r -a stat_fields <<< "$stat_rest"
    ((${#stat_fields[@]} >= 20)) || return 1

    PROCESS_STATE=${stat_fields[0]}
    PROCESS_START_TIME=${stat_fields[19]}
}

process_is_same_and_alive() {
    local pid=$1
    local expected_start_time=$2

    read_process_identity "$pid" || return 1

    if [[ "$PROCESS_START_TIME" != "$expected_start_time" ]]; then
        log "PID ${pid} was reused; the original process is considered finished."
        return 1
    fi

    [[ "$PROCESS_STATE" != "Z" && "$PROCESS_STATE" != "X" ]]
}

declare -a pids=()
declare -a start_times=()
declare -a finished=()
interval=30
settle_seconds=30

while (($# > 0)); do
    case "$1" in
        --pid)
            (($# >= 2)) || die "--pid requires a value"
            pids+=("$2")
            shift 2
            ;;
        --interval)
            (($# >= 2)) || die "--interval requires a value"
            interval=$2
            shift 2
            ;;
        --settle-seconds)
            (($# >= 2)) || die "--settle-seconds requires a value"
            settle_seconds=$2
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

((${#pids[@]} > 0)) || die "at least one --pid is required"
(($# > 0)) || die "a command is required after --"
is_nonnegative_integer "$interval" && ((interval > 0)) || \
    die "--interval must be a positive integer"
is_nonnegative_integer "$settle_seconds" || \
    die "--settle-seconds must be a non-negative integer"

for pid in "${pids[@]}"; do
    is_nonnegative_integer "$pid" && ((pid > 0)) || die "invalid PID: $pid"
    [[ "$pid" != "$$" ]] || die "cannot wait for this script's own PID ($$)"

    read_process_identity "$pid" || \
        die "PID ${pid} is not running; refusing to launch the command"
    [[ "$PROCESS_STATE" != "Z" && "$PROCESS_STATE" != "X" ]] || \
        die "PID ${pid} is already a zombie; refusing to launch the command"

    start_times+=("$PROCESS_START_TIME")
    finished+=(0)

    command_line=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)
    log "Watching PID ${pid}: ${command_line:-<command line unavailable>}"
done

printf -v quoted_command '%q ' "$@"
log "Queued command: ${quoted_command% }"

trap 'log "Waiter interrupted; queued command was not started."; exit 130' INT
trap 'log "Waiter terminated; queued command was not started."; exit 143' TERM HUP

while true; do
    remaining=0

    for index in "${!pids[@]}"; do
        [[ "${finished[$index]}" == 0 ]] || continue

        if process_is_same_and_alive \
            "${pids[$index]}" "${start_times[$index]}"; then
            ((remaining += 1))
        else
            finished[$index]=1
            log "PID ${pids[$index]} has finished."
        fi
    done

    ((remaining > 0)) || break
    log "Waiting for ${remaining} process(es); checking again in ${interval}s."
    sleep "$interval"
done

if ((settle_seconds > 0)); then
    log "All watched processes finished; waiting ${settle_seconds}s for resource cleanup."
    sleep "$settle_seconds"
fi

log "Starting queued command."
exec "$@"
