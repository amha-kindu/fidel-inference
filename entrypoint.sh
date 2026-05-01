#!/usr/bin/env bash
set -e
# optional: run DB migrations, model warmup, etc.
echo "Starting server..."
exec "$@"