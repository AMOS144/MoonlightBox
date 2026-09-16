#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="${MOONLIGHTBOX_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# 默认把运行数据放在所有 worktree 共享的目录，避免切换分支后创建一份空 SQLite。
# 部署或测试可以通过这些环境变量显式覆盖。
RUNTIME_DATA_DIR="${MOONLIGHTBOX_DATA_DIR:-$ROOT_DIR/../.runtime-data/data}"
export MOONLIGHTBOX_DATA_DIR="$RUNTIME_DATA_DIR"
export MOONLIGHTBOX_DATABASE_URL="${MOONLIGHTBOX_DATABASE_URL:-sqlite:///$RUNTIME_DATA_DIR/moonlightbox.db}"
export MOONLIGHTBOX_CHROMA_DIR="${MOONLIGHTBOX_CHROMA_DIR:-$RUNTIME_DATA_DIR/chroma}"
BACKEND_PID=""
FRONTEND_PID=""
PERSONA_RUNTIME_PID=""
REALTIME_WORKER_PID=""
COGNITION_WORKER_PID=""
BACKGROUND_WORKER_PID=""
TRAINING_WORKER_PID=""
PHOENIX_PID=""
SHUTDOWN_GRACE_STEPS="${MOONLIGHTBOX_SHUTDOWN_GRACE_STEPS:-50}"
SHUTDOWN_POLL_SECONDS="${MOONLIGHTBOX_SHUTDOWN_POLL_SECONDS:-0.1}"
MONITOR_POLL_SECONDS="${MOONLIGHTBOX_MONITOR_POLL_SECONDS:-1}"
BACKEND_PORT="${MOONLIGHTBOX_BACKEND_PORT:-8001}"
FRONTEND_PORT="${MOONLIGHTBOX_FRONTEND_PORT:-5175}"
PERSONA_RUNTIME_PORT="${PERSONA_RUNTIME_PORT:-8765}"
PHOENIX_PORT="${MOONLIGHTBOX_PHOENIX_PORT:-6006}"
PHOENIX_ENABLED="${MOONLIGHTBOX_PHOENIX_ENABLED:-true}"
# 标准启动脚本只连接本机 Collector；开发排障需要看到工具实参和模型结果，默认保留
# 脱敏且限长后的内容。Compose/远端环境仍在各自配置中默认关闭。
PHOENIX_CAPTURE_CONTENT="${MOONLIGHTBOX_PHOENIX_CAPTURE_CONTENT:-true}"
PHOENIX_DATA_DIR="${MOONLIGHTBOX_PHOENIX_DATA_DIR:-$ROOT_DIR/../.runtime-data/phoenix}"
PHOENIX_HEALTH_ATTEMPTS="${MOONLIGHTBOX_PHOENIX_HEALTH_ATTEMPTS:-120}"
PHOENIX_HEALTH_POLL_SECONDS="${MOONLIGHTBOX_PHOENIX_HEALTH_POLL_SECONDS:-0.5}"
PERSONA_RUNTIME_URL="${MOONLIGHTBOX_PERSONA_INFERENCE_URL:-http://127.0.0.1:${PERSONA_RUNTIME_PORT}}"
PERSONA_HEALTH_ATTEMPTS="${MOONLIGHTBOX_PERSONA_HEALTH_ATTEMPTS:-60}"
PERSONA_HEALTH_POLL_SECONDS="${MOONLIGHTBOX_PERSONA_HEALTH_POLL_SECONDS:-0.5}"
VENV_PYTHON=""
LAST_GROUP_PID=""

require_command() {
  local command_name="$1"
  local install_hint="$2"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "错误：未找到 ${command_name}。${install_hint}" >&2
    exit 1
  fi
}

ensure_port_available() {
  local port="$1"
  local service_name="$2"
  if command -v lsof >/dev/null 2>&1 &&
    lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "错误：${service_name}端口 ${port} 已被占用，请先关闭已有服务。" >&2
    exit 1
  fi
}

group_is_running() {
  local group_id="$1"
  [[ -n "$group_id" ]] && kill -0 -- "-$group_id" >/dev/null 2>&1
}

service_is_running() {
  local service_pid="$1"
  local process_state
  process_state="$(ps -o state= -p "$service_pid" 2>/dev/null || true)"
  [[ -n "$process_state" && "$process_state" != Z* ]]
}

signal_group() {
  local signal_name="$1"
  local group_id="$2"
  if group_is_running "$group_id"; then
    kill "-$signal_name" -- "-$group_id" >/dev/null 2>&1 || true
  fi
}

start_service_group() {
  local working_dir="$1"
  shift
  "$VENV_PYTHON" -c \
    'import os, sys; os.chdir(sys.argv[1]); os.setsid(); os.execvp(sys.argv[2], sys.argv[2:])' \
    "$working_dir" "$@" &
  LAST_GROUP_PID="$!"
}

cleanup() {
  trap - EXIT INT TERM
  for pid in "$BACKEND_PID" "$FRONTEND_PID" "$PERSONA_RUNTIME_PID" "$REALTIME_WORKER_PID" "$COGNITION_WORKER_PID" "$BACKGROUND_WORKER_PID" "$TRAINING_WORKER_PID" "$PHOENIX_PID"; do
    signal_group TERM "$pid"
  done
  local step
  for ((step = 0; step < SHUTDOWN_GRACE_STEPS; step++)); do
    local any_running=0
    for pid in "$BACKEND_PID" "$FRONTEND_PID" "$PERSONA_RUNTIME_PID" "$REALTIME_WORKER_PID" "$COGNITION_WORKER_PID" "$BACKGROUND_WORKER_PID" "$TRAINING_WORKER_PID" "$PHOENIX_PID"; do
      if group_is_running "$pid"; then
        any_running=1
      fi
    done
    if [[ "$any_running" -eq 0 ]]; then
      break
    fi
    sleep "$SHUTDOWN_POLL_SECONDS"
  done
  for pid in "$BACKEND_PID" "$FRONTEND_PID" "$PERSONA_RUNTIME_PID" "$REALTIME_WORKER_PID" "$COGNITION_WORKER_PID" "$BACKGROUND_WORKER_PID" "$TRAINING_WORKER_PID" "$PHOENIX_PID"; do
    signal_group KILL "$pid"
  done
  for pid in "$BACKEND_PID" "$FRONTEND_PID" "$PERSONA_RUNTIME_PID" "$REALTIME_WORKER_PID" "$COGNITION_WORKER_PID" "$BACKGROUND_WORKER_PID" "$TRAINING_WORKER_PID" "$PHOENIX_PID"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" >/dev/null 2>&1 || true
    fi
  done
}

handle_signal() {
  echo
  echo "正在关闭月光宝盒……"
  cleanup
  exit 130
}

trap cleanup EXIT
trap handle_signal INT TERM

cd "$ROOT_DIR"

require_command "uv" "请先安装：https://docs.astral.sh/uv/"
require_command "node" "请安装 Node.js 22 或更高版本。"
require_command "npm" "请安装 Node.js 自带的 npm。"
require_command "curl" "请先安装 curl。"
ensure_port_available "$BACKEND_PORT" "后端"
ensure_port_available "$FRONTEND_PORT" "前端"
if [[ "$PHOENIX_ENABLED" == "true" ]]; then
  ensure_port_available "$PHOENIX_PORT" "Phoenix"
fi
export MOONLIGHTBOX_BACKEND_PORT="$BACKEND_PORT"
export MOONLIGHTBOX_FRONTEND_PORT="$FRONTEND_PORT"
export MOONLIGHTBOX_PERSONA_INFERENCE_URL="$PERSONA_RUNTIME_URL"
export MOONLIGHTBOX_PHOENIX_ENABLED="$PHOENIX_ENABLED"
export MOONLIGHTBOX_PHOENIX_CAPTURE_CONTENT="$PHOENIX_CAPTURE_CONTENT"
if [[ "$PHOENIX_ENABLED" == "true" ]]; then
  export MOONLIGHTBOX_PHOENIX_COLLECTOR_ENDPOINT="http://127.0.0.1:${PHOENIX_PORT}/v1/traces"
fi

MANAGE_PERSONA_RUNTIME="${MOONLIGHTBOX_MANAGE_PERSONA_RUNTIME:-auto}"
if [[ "$MANAGE_PERSONA_RUNTIME" == "auto" ]]; then
  if [[ "$(uname -s)" == "Linux" ]]; then
    MANAGE_PERSONA_RUNTIME="true"
  else
    MANAGE_PERSONA_RUNTIME="false"
  fi
fi
if [[ "$MANAGE_PERSONA_RUNTIME" == "true" ]]; then
  ensure_port_available "$PERSONA_RUNTIME_PORT" "人格推理"
fi

if [[ ! -f ".env" ]]; then
  echo "提示：未找到 .env，将使用默认配置。可复制 .env.example 后按需修改。"
fi

UV_SYNC_ARGS=()
# 后端的 observability 模块在导入阶段提供 Phoenix no-op/路由契约；即使关闭
# Collector，也必须安装该可选依赖，否则 API 会在启动前因 phoenix.client 导入失败。
UV_SYNC_ARGS+=(--extra observability)
if [[ "$(uname -s)" == "Linux" ]]; then
  echo "正在同步 Python 与 Linux PyTorch/Transformers/PEFT 依赖……"
  UV_SYNC_ARGS+=(--extra linux-ml)
else
  echo "正在同步 Python 依赖（人格推理服务仅支持 Linux）……"
fi
uv sync "${UV_SYNC_ARGS[@]}"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "错误：uv sync 后未找到项目 Python：$VENV_PYTHON" >&2
  exit 1
fi

if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_MEMORY_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9')"
  if [[ -n "$GPU_MEMORY_MB" && "$GPU_MEMORY_MB" -lt 10240 ]]; then
    echo "提示：检测到 ${GPU_MEMORY_MB}MB GPU 显存。历史 Qwen3-8B LoRA 无法在此显卡加载；"
    echo "      请转换后放到至少 10GB 的 Linux GPU，或使用默认 Qwen3-1.7B 重新训练。"
  fi
fi

echo "正在初始化本地数据目录：$RUNTIME_DATA_DIR……"
mkdir -p "$RUNTIME_DATA_DIR" models
PYTHONPATH=backend uv run python -c \
  "from moonlightbox.config import Settings; Settings().ensure_directories()"

if [[ "$PHOENIX_ENABLED" == "true" ]]; then
  # Phoenix 首次启动需要迁移自身的数据。先并行启动它，把耗时隐藏在前端依赖和业务迁移中；
  # 在启动 API/Worker 前再完成就绪检查，避免第一批 Agent Run 因 Collector 未就绪而丢失。
  echo "正在启动 Phoenix Trace 浏览器：http://127.0.0.1:${PHOENIX_PORT}"
  mkdir -p "$PHOENIX_DATA_DIR"
  start_service_group "$ROOT_DIR" \
    env PHOENIX_WORKING_DIR="$PHOENIX_DATA_DIR" PHOENIX_DISABLE_AGENT_ASSISTANT=true \
    PHOENIX_LOG_MIGRATIONS=false PHOENIX_DB_LOGGING_LEVEL=warning \
    uv run --extra observability phoenix serve \
    --host 127.0.0.1 --port "$PHOENIX_PORT"
  PHOENIX_PID="$LAST_GROUP_PID"
fi

echo "正在安装前端依赖……"
(
  cd frontend
  npm install
)

echo "正在升级数据库……"
uv run alembic -c backend/alembic.ini upgrade head

if [[ "$PHOENIX_ENABLED" == "true" ]]; then
  PHOENIX_HEALTH_URL="http://127.0.0.1:${PHOENIX_PORT}"
  for ((attempt = 1; attempt <= PHOENIX_HEALTH_ATTEMPTS; attempt++)); do
    if curl --fail --silent --show-error "$PHOENIX_HEALTH_URL" >/dev/null; then
      break
    fi
    if ! service_is_running "$PHOENIX_PID"; then
      echo "错误：Phoenix Trace 浏览器在就绪检查前退出。" >&2
      exit 1
    fi
    if [[ "$attempt" -eq "$PHOENIX_HEALTH_ATTEMPTS" ]]; then
      echo "错误：Phoenix Trace 浏览器启动超时，请查看其进程输出。" >&2
      exit 1
    fi
    sleep "$PHOENIX_HEALTH_POLL_SECONDS"
  done
fi

echo "正在启动后端：http://127.0.0.1:${BACKEND_PORT}"
start_service_group "$ROOT_DIR" \
  uv run uvicorn moonlightbox.api:app --app-dir backend --port "$BACKEND_PORT"
BACKEND_PID="$LAST_GROUP_PID"

echo "正在启动前端：http://localhost:${FRONTEND_PORT}"
start_service_group "$ROOT_DIR/frontend" \
  npm run dev -- --strictPort --port "$FRONTEND_PORT"
FRONTEND_PID="$LAST_GROUP_PID"

if [[ "$MANAGE_PERSONA_RUNTIME" == "true" ]]; then
  export PERSONA_RUNTIME_PORT
  echo "正在启动 Linux 人格推理运行时：http://127.0.0.1:${PERSONA_RUNTIME_PORT}"
  start_service_group "$ROOT_DIR" \
    uv run uvicorn moonlightbox.persona_runtime:app --app-dir backend --port "$PERSONA_RUNTIME_PORT"
  PERSONA_RUNTIME_PID="$LAST_GROUP_PID"
else
  echo "未托管本地人格服务；检查远端 Linux 推理服务：${PERSONA_RUNTIME_URL}"
fi

if [[ "$MANAGE_PERSONA_RUNTIME" == "true" ]]; then
  HEALTH_URL="http://127.0.0.1:${PERSONA_RUNTIME_PORT}/health"
else
  HEALTH_URL="${PERSONA_RUNTIME_URL%/}/health"
fi
for ((attempt = 1; attempt <= PERSONA_HEALTH_ATTEMPTS; attempt++)); do
  if curl --fail --silent --show-error "$HEALTH_URL" >/dev/null; then
    break
  fi
  if [[ "$MANAGE_PERSONA_RUNTIME" == "true" ]] && ! service_is_running "$PERSONA_RUNTIME_PID"; then
    echo "错误：人格推理运行时在健康检查前退出。" >&2
    exit 1
  fi
  if [[ "$attempt" -eq "$PERSONA_HEALTH_ATTEMPTS" ]]; then
    echo "错误：Linux 人格推理服务健康检查超时，请确认 ${PERSONA_RUNTIME_URL} 可达。" >&2
    exit 1
  fi
  sleep "$PERSONA_HEALTH_POLL_SECONDS"
done

echo "正在启动实时会话 Worker"
PYTHONPATH=backend start_service_group \
  "$ROOT_DIR" env MOONLIGHTBOX_WORKER_ROLE=realtime uv run python -m moonlightbox.worker_main
REALTIME_WORKER_PID="$LAST_GROUP_PID"

echo "正在启动离线认知 Worker"
PYTHONPATH=backend start_service_group \
  "$ROOT_DIR" env MOONLIGHTBOX_WORKER_ROLE=cognition uv run python -m moonlightbox.worker_main
COGNITION_WORKER_PID="$LAST_GROUP_PID"

echo "正在启动后台记忆 Worker"
PYTHONPATH=backend start_service_group \
  "$ROOT_DIR" env MOONLIGHTBOX_WORKER_ROLE=background uv run python -m moonlightbox.worker_main
BACKGROUND_WORKER_PID="$LAST_GROUP_PID"

echo "正在启动 Linux GPU 训练 Worker"
PYTHONPATH=backend start_service_group \
  "$ROOT_DIR" env MOONLIGHTBOX_WORKER_ROLE=training uv run python -m moonlightbox.worker_main
TRAINING_WORKER_PID="$LAST_GROUP_PID"

echo "月光宝盒已启动。按 Ctrl+C 同时关闭前端、后端和 Worker。"

EXIT_STATUS=0
while true; do
  if ! service_is_running "$BACKEND_PID"; then
    wait "$BACKEND_PID" || EXIT_STATUS="$?"
    break
  fi
  if ! service_is_running "$FRONTEND_PID"; then
    wait "$FRONTEND_PID" || EXIT_STATUS="$?"
    break
  fi
  if [[ -n "$PERSONA_RUNTIME_PID" ]] && ! service_is_running "$PERSONA_RUNTIME_PID"; then
    wait "$PERSONA_RUNTIME_PID" || EXIT_STATUS="$?"
    break
  fi
  if ! service_is_running "$REALTIME_WORKER_PID"; then
    wait "$REALTIME_WORKER_PID" || EXIT_STATUS="$?"
    break
  fi
  if ! service_is_running "$COGNITION_WORKER_PID"; then
    wait "$COGNITION_WORKER_PID" || EXIT_STATUS="$?"
    break
  fi
  if ! service_is_running "$BACKGROUND_WORKER_PID"; then
    wait "$BACKGROUND_WORKER_PID" || EXIT_STATUS="$?"
    break
  fi
  if [[ -n "$PHOENIX_PID" ]] && ! service_is_running "$PHOENIX_PID"; then
    wait "$PHOENIX_PID" || EXIT_STATUS="$?"
    break
  fi
  if ! service_is_running "$TRAINING_WORKER_PID"; then
    wait "$TRAINING_WORKER_PID" || EXIT_STATUS="$?"
    break
  fi
  sleep "$MONITOR_POLL_SECONDS"
done

exit "$EXIT_STATUS"
