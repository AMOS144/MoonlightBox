#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p /app/data/lightrag /app/data/chroma /app/data/models

迁移失败() {
  local exit_code=$?
  echo "错误:数据库迁移失败,服务未启动。" >&2
  exit "$exit_code"
}

trap 迁移失败 ERR
/app/.venv/bin/alembic -c backend/alembic.ini upgrade head
trap - ERR

exec /usr/bin/supervisord -c /app/docker/supervisord.conf
