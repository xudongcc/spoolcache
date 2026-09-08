#!/usr/bin/env bash
# Run the pinned Gemma functional-development baseline across two DGX Sparks.
# This is an external development launcher, not a SpoolCache model profile or
# process supervisor. It intentionally keeps the qualified model/runtime/PP
# tuple fixed while discovering cache semantics through vLLM at runtime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${GEMMA_PP2_ENV_FILE:-$PROJECT_ROOT/.env.gemma-pp2}"

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

# Reproducible development baseline. Change these only together with a new
# no-cache correctness qualification; never copy them into connector logic.
readonly MODEL_ID="google/gemma-4-E2B-it"
readonly MODEL_REVISION="3e22461f65e89153144f8adb70e3b8c2cc9845a7"
readonly IMAGE="spoolcache-dev:vllm-0.28.0"
readonly PP_LAYER_PARTITION="12,23"
readonly API_PORT="8000"
readonly MASTER_PORT="29532"
readonly HEAD_CONTAINER="spoolcache-gemma-pp2-head"
readonly WORKER_CONTAINER="spoolcache-gemma-pp2-worker"
readonly CACHE_MAX_SIZE="200"
readonly CONTAINER_HF_HOME="/root/.cache/huggingface"
readonly CONTAINER_CACHE_ROOT="/var/lib/spoolcache"

# Host placement and wiring are the only supported overrides. Defaults match
# the current two-node Spark lab and route SSH/model/image traffic over CX-7.
GEMMA_HEAD_IP="${GEMMA_HEAD_IP:-10.100.216.1}"
GEMMA_WORKER_IP="${GEMMA_WORKER_IP:-10.100.216.2}"
GEMMA_WORKER_SSH="${GEMMA_WORKER_SSH:-$GEMMA_WORKER_IP}"
GEMMA_HEAD_HF_HOME="${GEMMA_HEAD_HF_HOME:-/root/.cache/huggingface}"
GEMMA_WORKER_HF_HOME="${GEMMA_WORKER_HF_HOME:-/root/.cache/huggingface}"
SPOOLCACHE_PATH="${SPOOLCACHE_PATH-~/.cache/spoolcache}"
GEMMA_HEAD_GLOO_IFACE="${GEMMA_HEAD_GLOO_IFACE:-enp1s0f1np1}"
GEMMA_WORKER_GLOO_IFACE="${GEMMA_WORKER_GLOO_IFACE:-enp1s0f1np1}"
GEMMA_HEAD_NCCL_IFACES="${GEMMA_HEAD_NCCL_IFACES:-=enp1s0f1np1,enP2p1s0f1np1}"
GEMMA_WORKER_NCCL_IFACES="${GEMMA_WORKER_NCCL_IFACES:-=enp1s0f1np1,enP2p1s0f1np1}"
GEMMA_HEAD_NCCL_HCAS="${GEMMA_HEAD_NCCL_HCAS:-=rocep1s0f1,roceP2p1s0f1}"
GEMMA_WORKER_NCCL_HCAS="${GEMMA_WORKER_NCCL_HCAS:-=rocep1s0f1,roceP2p1s0f1}"
GEMMA_NCCL_GID_INDEX="${GEMMA_NCCL_GID_INDEX:-3}"

readonly MODEL_REPO_REL="hub/models--google--gemma-4-E2B-it"
readonly MODEL_SNAPSHOT_REL="$MODEL_REPO_REL/snapshots/$MODEL_REVISION"

STARTING=0
RUN_IMAGE_ID=""

info() { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
ok() { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: scripts/gemma-pp2-dev.sh COMMAND [TARGET]

Commands:
  preflight       Check both hosts, GPUs, CX-7, image and pinned model
  image-sync      Stream the local development image to the worker over CX-7
  model-sync      Rsync the pinned Hugging Face repository cache over CX-7
  start           Preflight, start worker then head, wait for API
  stop            Stop and remove the worker/head development containers
  restart         Stop, then start the complete PP group
  status          Print container and API state as one JSON object
  logs TARGET     Follow logs for head or worker (default: head)

Optional host overrides are read from:
  $ENV_FILE

The model, revision, image, TP=1/PP=2 layout and cache policy are deliberately
fixed. Copy scripts/gemma-pp2-dev.env.example only to change lab wiring/paths.
EOF
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command is missing: $1"
}

remote_exec() {
    local remote_command=""
    printf -v remote_command '%q ' "$@"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$GEMMA_WORKER_SSH" "$remote_command"
}

# Resolve the same setting against each host's home before building Docker args.
spoolcache_host_path() {
    local value="$1" user_home="$2"
    case "$value" in
        "~/"*) value="$user_home/${value:2}" ;;
    esac
    # These launchers embed bind mounts in shell command strings.
    [[ "$value" =~ ^/[a-zA-Z0-9_./-]+$ && "$value" != "/" ]] || {
        printf '%s\n' "SPOOLCACHE_PATH must be an absolute path without spaces or shell metacharacters" >&2
        return 2
    }
    printf '%s\n' "$value"
}

validate_host_path() {
    local label="$1"
    local value="$2"
    [[ "$value" == /* && "$value" != "/" && "$value" != *:* \
        && "$value" != *[[:space:]]* ]] \
        || fail "$label must be an absolute path without whitespace or ':'"
}

validate_configuration() {
    [[ "$GEMMA_WORKER_SSH" =~ ^[A-Za-z0-9._@:-]+$ && "$GEMMA_WORKER_SSH" != -* ]] \
        || fail "GEMMA_WORKER_SSH must be one SSH host or user@host"
    [[ "$GEMMA_NCCL_GID_INDEX" =~ ^[0-9]+$ ]] \
        || fail "GEMMA_NCCL_GID_INDEX must be a non-negative integer"
    validate_host_path GEMMA_HEAD_HF_HOME "$GEMMA_HEAD_HF_HOME"
    validate_host_path GEMMA_WORKER_HF_HOME "$GEMMA_WORKER_HF_HOME"
    spoolcache_host_path "$SPOOLCACHE_PATH" "$HOME" >/dev/null
}

container_state_local() {
    local state
    if ! state="$(docker inspect --format '{{.State.Status}}' "$1" 2>/dev/null)" \
        || [[ -z "$state" ]]; then
        state="absent"
    fi
    printf '%s\n' "$state"
}

container_state_remote() {
    local state
    if ! state="$(
        remote_exec docker inspect --format '{{.State.Status}}' "$1" 2>/dev/null
    )" || [[ -z "$state" ]]; then
        state="absent"
    fi
    printf '%s\n' "$state"
}

ensure_not_running() {
    [[ "$(container_state_local "$HEAD_CONTAINER")" != "running" ]] \
        || fail "$HEAD_CONTAINER is already running; use restart"
    [[ "$(container_state_remote "$WORKER_CONTAINER")" != "running" ]] \
        || fail "$WORKER_CONTAINER is already running; use restart"
}

check_gpu_idle() {
    local head_tenants worker_tenants
    head_tenants="$(
        nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d'
    )"
    worker_tenants="$(
        remote_exec nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d'
    )"
    if [[ -n "$head_tenants" || -n "$worker_tenants" ]]; then
        printf 'head GPU users: %s\n' "${head_tenants:-idle}" >&2
        printf 'worker GPU users: %s\n' "${worker_tenants:-idle}" >&2
        fail "a GPU is in use; stop the active DeepSeek/Qwen/GLM/Compose service first"
    fi
}

check_cx7_selectors() {
    local item
    local -a head_ifaces worker_ifaces head_hcas worker_hcas
    IFS=',' read -r -a head_ifaces <<< "${GEMMA_HEAD_NCCL_IFACES#=}"
    IFS=',' read -r -a worker_ifaces <<< "${GEMMA_WORKER_NCCL_IFACES#=}"
    IFS=',' read -r -a head_hcas <<< "${GEMMA_HEAD_NCCL_HCAS#=}"
    IFS=',' read -r -a worker_hcas <<< "${GEMMA_WORKER_NCCL_HCAS#=}"
    for item in "${head_ifaces[@]}"; do
        ip link show dev "$item" >/dev/null 2>&1 \
            || fail "head NCCL interface is missing: $item"
    done
    for item in "${worker_ifaces[@]}"; do
        remote_exec ip link show dev "$item" >/dev/null \
            || fail "worker NCCL interface is missing: $item"
    done
    for item in "${head_hcas[@]}"; do
        [[ -d "/sys/class/infiniband/$item" ]] \
            || fail "head NCCL HCA is missing: $item"
    done
    for item in "${worker_hcas[@]}"; do
        remote_exec test -d "/sys/class/infiniband/$item" \
            || fail "worker NCCL HCA is missing: $item"
    done
    ip -4 route get "$GEMMA_WORKER_IP" | grep -Fq "dev $GEMMA_HEAD_GLOO_IFACE" \
        || fail "head route to worker does not use $GEMMA_HEAD_GLOO_IFACE"
    remote_exec ip -4 route get "$GEMMA_HEAD_IP" \
        | grep -Fq "dev $GEMMA_WORKER_GLOO_IFACE" \
        || fail "worker route to head does not use $GEMMA_WORKER_GLOO_IFACE"
}

preflight() {
    validate_configuration
    for command_name in docker ssh tar sha256sum curl ss nvidia-smi git; do
        require_command "$command_name"
    done
    remote_exec true >/dev/null
    for command_name in docker tar nvidia-smi ip mkdir; do
        remote_exec command -v "$command_name" >/dev/null \
            || fail "worker command is missing: $command_name"
    done
    [[ -d /dev/infiniband ]] || fail "head /dev/infiniband is missing"
    remote_exec test -d /dev/infiniband || fail "worker /dev/infiniband is missing"
    ip -4 addr show dev "$GEMMA_HEAD_GLOO_IFACE" | grep -Fq "$GEMMA_HEAD_IP/" \
        || fail "head IP $GEMMA_HEAD_IP is not assigned to $GEMMA_HEAD_GLOO_IFACE"
    remote_exec ip -4 addr show dev "$GEMMA_WORKER_GLOO_IFACE" \
        | grep -Fq "$GEMMA_WORKER_IP/" \
        || fail "worker IP $GEMMA_WORKER_IP is not assigned to $GEMMA_WORKER_GLOO_IFACE"
    check_cx7_selectors
    [[ -d "$GEMMA_HEAD_HF_HOME/$MODEL_SNAPSHOT_REL" ]] \
        || fail "pinned model is missing on head; run hf download, then model-sync"
    remote_exec test -d "$GEMMA_WORKER_HF_HOME/$MODEL_SNAPSHOT_REL" \
        || fail "pinned model is missing on worker; run model-sync"
    local head_image_id worker_image_id
    head_image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
    worker_image_id="$(
        remote_exec docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true
    )"
    [[ -n "$head_image_id" ]] || fail "local image $IMAGE is missing; run docker compose build"
    [[ "$head_image_id" == "$worker_image_id" ]] \
        || fail "worker image differs or is missing; run image-sync"
    local wheel_digest
    wheel_digest="$(docker image inspect --format '{{index .Config.Labels "io.spoolcache.wheel.sha256"}}' "$head_image_id")"
    [[ "$wheel_digest" =~ ^[0-9a-f]{64}$ ]] \
        || fail "development image must contain an authenticated release wheel"
    # Resolve the mutable convenience tag once for the complete PP group.
    RUN_IMAGE_ID="$head_image_id"
    ensure_not_running
    if ss -H -ltn "sport = :$API_PORT" | grep -q .; then
        fail "head TCP port $API_PORT is already in use"
    fi
    check_gpu_idle
    ok "two-node preflight passed (image $head_image_id)"
}

image_sync() {
    validate_configuration
    require_command docker
    require_command ssh
    docker image inspect "$IMAGE" >/dev/null 2>&1 \
        || fail "local image $IMAGE is missing; run docker compose build"
    remote_exec docker info >/dev/null \
        || fail "worker Docker daemon is unavailable"
    info "streaming $IMAGE to $GEMMA_WORKER_SSH over the CX-7 SSH path"
    docker save "$IMAGE" | remote_exec docker load
    local head_image_id worker_image_id
    head_image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
    worker_image_id="$(remote_exec docker image inspect --format '{{.Id}}' "$IMAGE")"
    [[ "$head_image_id" == "$worker_image_id" ]] \
        || fail "image ID mismatch after transfer"
    ok "worker image matches head: $head_image_id"
}

model_sync() {
    validate_configuration
    require_command rsync
    require_command ssh
    local head_repo="$GEMMA_HEAD_HF_HOME/$MODEL_REPO_REL"
    local worker_repo="$GEMMA_WORKER_HF_HOME/$MODEL_REPO_REL"
    remote_exec true >/dev/null || fail "worker SSH is unavailable"
    remote_exec command -v rsync >/dev/null \
        || fail "worker command is missing: rsync"
    [[ -d "$head_repo/snapshots/$MODEL_REVISION" ]] \
        || fail "pinned model is missing: $head_repo/snapshots/$MODEL_REVISION"
    remote_exec mkdir -p "$GEMMA_WORKER_HF_HOME/hub"
    info "rsyncing the Gemma repository cache to $GEMMA_WORKER_SSH over CX-7"
    rsync -a --partial --info=progress2 -e \
        'ssh -o BatchMode=yes -o ConnectTimeout=10' \
        "$head_repo/" "$GEMMA_WORKER_SSH:$worker_repo/"
    remote_exec test -d "$worker_repo/snapshots/$MODEL_REVISION" \
        || fail "worker does not contain the pinned snapshot after rsync"
    ok "worker has $MODEL_ID@$MODEL_REVISION"
}

cleanup_failed_start() {
    local status=$?
    if [[ "$STARTING" == "1" ]]; then
        printf '\n' >&2
        info "startup did not reach readiness; removing the incomplete PP group"
        remote_exec docker rm -f "$WORKER_CONTAINER" >/dev/null 2>&1 || true
        docker rm -f "$HEAD_CONTAINER" >/dev/null 2>&1 || true
    fi
    return "$status"
}

docker_args() {
    local node_rank="$1"
    local host_ip="$2"
    local gloo_iface="$3"
    local nccl_ifaces="$4"
    local nccl_hcas="$5"
    local hf_home="$6"
    local cache_root="$7"
    local container_name="$8"
    local -n output="$9"
    local kv_config
    kv_config="{\"kv_connector\":\"SpoolCacheConnector\",\"kv_role\":\"kv_both\",\"kv_connector_module_path\":\"spoolcache.vllm.connector\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"spoolcache_path\":\"$CONTAINER_CACHE_ROOT\",\"spoolcache_max_size\":$CACHE_MAX_SIZE}}"
    output=(
        docker run -d --name "$container_name" --init
        --gpus all --ipc host --network host --stop-timeout 120
        --cap-add IPC_LOCK --ulimit memlock=-1
        --device /dev/infiniband:/dev/infiniband
        -e "HF_HUB_OFFLINE=1"
        -e "VLLM_LOGGING_LEVEL=DEBUG"
        -e "VLLM_SERVER_DEV_MODE=1"
        -e "VLLM_PP_LAYER_PARTITION=$PP_LAYER_PARTITION"
        -e "VLLM_HOST_IP=$host_ip"
        -e "GLOO_SOCKET_IFNAME=$gloo_iface"
        -e "NCCL_SOCKET_IFNAME=$nccl_ifaces"
        -e "NCCL_IB_HCA=$nccl_hcas"
        -e "NCCL_IB_GID_INDEX=$GEMMA_NCCL_GID_INDEX"
        -e "NCCL_IB_ADDR_FAMILY=AF_INET"
        -e "NCCL_IB_ROCE_VERSION_NUM=2"
        -e "NCCL_IB_DISABLE=0"
        -e "NCCL_CROSS_NIC=1"
        -e "NCCL_DEBUG=INFO"
        -e "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
        -v "$hf_home:$CONTAINER_HF_HOME:ro"
        -v "$cache_root:$CONTAINER_CACHE_ROOT"
        "${RUN_IMAGE_ID:?preflight must resolve the release image}"
        "$MODEL_ID"
        --revision "$MODEL_REVISION"
        --served-model-name "$MODEL_ID"
        --tensor-parallel-size 1
        --pipeline-parallel-size 2
        --distributed-executor-backend mp
        --data-parallel-backend mp
        --nnodes 2
        --node-rank "$node_rank"
        --master-addr "$GEMMA_HEAD_IP"
        --master-port "$MASTER_PORT"
        --max-model-len 8192
        --gpu-memory-utilization 0.80
        --max-num-seqs 1
        --enable-prefix-caching
        --enable-prompt-tokens-details
        --enforce-eager
        --kv-transfer-config "$kv_config"
    )
}

start_group() {
    preflight
    local head_cache_path worker_cache_path worker_home
    head_cache_path="$(spoolcache_host_path "$SPOOLCACHE_PATH" "$HOME")"
    worker_home="$(remote_exec bash -c 'printf "%s" "$HOME"')"
    worker_cache_path="$(spoolcache_host_path "$SPOOLCACHE_PATH" "$worker_home")"
    mkdir -p "$head_cache_path"
    remote_exec mkdir -p "$worker_cache_path"
    docker rm -f "$HEAD_CONTAINER" >/dev/null 2>&1 || true
    remote_exec docker rm -f "$WORKER_CONTAINER" >/dev/null 2>&1 || true

    local -a worker_args head_args
    docker_args 1 "$GEMMA_WORKER_IP" "$GEMMA_WORKER_GLOO_IFACE" \
        "$GEMMA_WORKER_NCCL_IFACES" "$GEMMA_WORKER_NCCL_HCAS" \
        "$GEMMA_WORKER_HF_HOME" "$worker_cache_path" \
        "$WORKER_CONTAINER" worker_args
    worker_args+=(--headless)
    docker_args 0 "$GEMMA_HEAD_IP" "$GEMMA_HEAD_GLOO_IFACE" \
        "$GEMMA_HEAD_NCCL_IFACES" "$GEMMA_HEAD_NCCL_HCAS" \
        "$GEMMA_HEAD_HF_HOME" "$head_cache_path" \
        "$HEAD_CONTAINER" head_args
    head_args+=(--host 0.0.0.0 --port "$API_PORT")

    STARTING=1
    trap cleanup_failed_start EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    info "starting worker PP stage first"
    remote_exec "${worker_args[@]}" >/dev/null
    info "starting head PP stage"
    "${head_args[@]}" >/dev/null

    local deadline=$((SECONDS + 900))
    while (( SECONDS < deadline )); do
        if [[ "$(container_state_local "$HEAD_CONTAINER")" != "running" ]]; then
            docker logs --tail 120 "$HEAD_CONTAINER" >&2 || true
            fail "head container exited before readiness"
        fi
        if [[ "$(container_state_remote "$WORKER_CONTAINER")" != "running" ]]; then
            remote_exec docker logs --tail 120 "$WORKER_CONTAINER" >&2 || true
            fail "worker container exited before readiness"
        fi
        if curl --fail --silent --show-error \
            "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
            STARTING=0
            trap - EXIT INT TERM
            ok "Gemma PP=2 API is ready at http://127.0.0.1:$API_PORT"
            return
        fi
        info "waiting for Gemma PP=2 API readiness"
        sleep 10
    done
    fail "API did not become ready within 900 seconds"
}

stop_group() {
    validate_configuration
    require_command docker
    require_command ssh
    docker info >/dev/null 2>&1 || fail "head Docker daemon is unavailable"
    remote_exec docker info >/dev/null \
        || fail "worker Docker daemon is unavailable; complete group was not stopped"
    info "stopping worker PP stage"
    if [[ "$(container_state_remote "$WORKER_CONTAINER")" != "absent" ]]; then
        remote_exec docker rm -f "$WORKER_CONTAINER" >/dev/null
    fi
    info "stopping head PP stage"
    if [[ "$(container_state_local "$HEAD_CONTAINER")" != "absent" ]]; then
        docker rm -f "$HEAD_CONTAINER" >/dev/null
    fi
    ok "Gemma PP=2 development group is stopped; cache roots were retained"
}

status_group() {
    validate_configuration
    require_command docker
    require_command ssh
    require_command curl
    local head_state worker_state api_ready=false
    if docker info >/dev/null 2>&1; then
        head_state="$(container_state_local "$HEAD_CONTAINER")"
    else
        head_state="unreachable"
    fi
    if remote_exec docker info >/dev/null 2>&1; then
        worker_state="$(container_state_remote "$WORKER_CONTAINER")"
    else
        worker_state="unreachable"
    fi
    if curl --fail --silent "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
        api_ready=true
    fi
    printf '{"head":"%s","worker":"%s","api_ready":%s,"api":"http://127.0.0.1:%s"}\n' \
        "$head_state" "$worker_state" "$api_ready" "$API_PORT"
}

show_logs() {
    validate_configuration
    local target="${1:-head}"
    case "$target" in
        head) docker logs --tail 200 --follow "$HEAD_CONTAINER" ;;
        worker) remote_exec docker logs --tail 200 --follow "$WORKER_CONTAINER" ;;
        *) fail "logs target must be head or worker" ;;
    esac
}

main() {
    local command="${1:-help}"
    shift || true
    case "$command" in
        preflight) preflight ;;
        image-sync) image_sync ;;
        model-sync) model_sync ;;
        start) start_group ;;
        stop) stop_group ;;
        restart) stop_group; start_group ;;
        status) status_group ;;
        logs) show_logs "${1:-head}" ;;
        help|-h|--help) usage ;;
        *) usage >&2; fail "unknown command: $command" ;;
    esac
}

main "$@"
