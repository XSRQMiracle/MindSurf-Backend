#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 || "$1" != ghcr.io/*:* ]]; then
    echo "usage: $0 ghcr.io/owner/image:immutable-tag" >&2
    exit 2
fi

image="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
compose_file="$script_dir/compose.production.yml"
state_file="$script_dir/.image.env"
lock_file="$script_dir/.deploy.lock"
port="${MINDSURF_PORT:-8000}"
health_url="http://127.0.0.1:${port}/health/ready"
health_attempts="${MINDSURF_HEALTH_ATTEMPTS:-120}"

exec 9>"$lock_file"
if ! flock -n 9; then
    echo "another MindSurf deployment is already running" >&2
    exit 1
fi

command -v docker >/dev/null
command -v curl >/dev/null
docker compose version >/dev/null

runtime_env="${MINDSURF_ENV_FILE:-/etc/mindsurf/backend.env}"
if [[ ! -r "$runtime_env" ]]; then
    echo "runtime environment file is missing or unreadable: $runtime_env" >&2
    exit 1
fi

previous_image=""
if [[ -f "$state_file" ]]; then
    previous_image="$(sed -n 's/^MINDSURF_IMAGE=//p' "$state_file" | tail -n 1)"
fi

write_state() {
    local selected_image="$1"
    local temporary
    temporary="${state_file}.tmp"
    printf 'MINDSURF_IMAGE=%s\n' "$selected_image" >"$temporary"
    mv -f -- "$temporary" "$state_file"
}

compose() {
    docker compose --env-file "$state_file" -f "$compose_file" "$@"
}

wait_until_ready() {
    local attempt
    for ((attempt = 1; attempt <= health_attempts; attempt++)); do
        if curl --fail --silent --show-error --max-time 5 "$health_url" >/dev/null; then
            return 0
        fi
        sleep 5
    done
    return 1
}

echo "pulling $image"
docker pull "$image"
write_state "$image"

echo "starting $image"
if compose up --detach --remove-orphans && wait_until_ready; then
    echo "deployment ready: $image"
    docker image prune --force --filter "until=168h" >/dev/null
    exit 0
fi

echo "deployment failed readiness check: $image" >&2
compose ps >&2 || true
compose logs --tail 200 backend >&2 || true

if [[ -z "$previous_image" || "$previous_image" == "$image" ]]; then
    echo "no previous immutable image is available for rollback" >&2
    rm -f -- "$state_file"
    exit 1
fi

echo "rolling back to $previous_image" >&2
docker pull "$previous_image"
write_state "$previous_image"
compose up --detach --remove-orphans

if wait_until_ready; then
    echo "rollback ready: $previous_image" >&2
else
    echo "rollback also failed readiness: $previous_image" >&2
    compose logs --tail 200 backend >&2 || true
fi

exit 1
