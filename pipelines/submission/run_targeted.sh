#!/usr/bin/env bash
# Small, explicitly authorized review. Existing checkpoints remain immutable.
set -Eeuo pipefail
cd /opt/ml/code
python -m pip install --disable-pip-version-check --no-input -r pipelines/oracle_reader/requirements-gpu.txt
export OMP_THREAD_LIMIT=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/opt/ml/code:/opt/ml/code/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -u -m pipelines.submission.targeted
