#!/usr/bin/env bash
set -Eeuo pipefail

迁移失败() {
  local exit_code=$?
  echo "错误：数据库迁移失败，服务未启动。" >&2
  exit "$exit_code"
}

trap 迁移失败 ERR
alembic -c backend/alembic.ini upgrade head
trap - ERR

if (( $# == 0 )); then
  echo "错误：数据库迁移成功，但没有提供要启动的服务命令。" >&2
  exit 64
fi

exec "$@"
