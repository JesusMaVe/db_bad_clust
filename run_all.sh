#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Iniciando Oracle 23c ==="
docker compose up -d

echo "=== Esperando a que Oracle esté listo ==="
for i in $(seq 1 60); do
    if docker compose exec -T oracle healthcheck.sh 2>/dev/null; then
        echo "Oracle listo!"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Oracle no arrancó en 60s"
        exit 1
    fi
    sleep 5
done

echo "=== Ejecutando generación de datos malos ==="
cd scripts
python orchestrator.py

echo "=== Listo! ==="
