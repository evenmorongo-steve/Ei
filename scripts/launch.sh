#!/usr/bin/env bash
# =====================================================================================
# Ei launcher — auto-detects the compute environment and submits accordingly.
#
#   ./scripts/launch.sh configs/experiment.yaml              # auto-detect
#   ./scripts/launch.sh configs/experiment.yaml --local      # force local
#   SCHEDULER=slurm ./scripts/launch.sh configs/experiment.yaml
#
# Detection order is deliberate: Slurm and Kubernetes are checked before Ray because a
# Ray cluster is frequently launched INSIDE a Slurm allocation, and in that case the
# Slurm path is the correct one (it owns the resource reservation).
#
# Every path runs the SAME `ei run` entry point with the SAME config, so the
# pre-registration hash is identical across environments and results are comparable.
# =====================================================================================
set -Eeuo pipefail

CONFIG="${1:-configs/experiment.yaml}"
shift || true
FORCE=""
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --slurm) FORCE="slurm" ;;
    --k8s|--kubernetes) FORCE="k8s" ;;
    --ray) FORCE="ray" ;;
    --local) FORCE="local" ;;
    --dry-run) DRY_RUN=1 ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$CONFIG" ]]; then
  echo "FATAL: config not found: $CONFIG" >&2
  exit 2
fi

# -------------------------------------------------------------------------------------
# Resolve the interpreter.  A local .venv is preferred over whatever python happens to be
# on PATH, because a silently different environment is a reproducibility failure.
# -------------------------------------------------------------------------------------
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "FATAL: no python3 found" >&2
  exit 2
fi

# -------------------------------------------------------------------------------------
# Gate 1: config must be schema-valid and the design must not have blocking gaps.
# Failing fast here is the point -- a week of GPU time on an unfalsifiable design is the
# expensive failure this project exists to prevent.
# -------------------------------------------------------------------------------------
echo "==> validating $CONFIG"
if ! "$PY" -m ei.cli validate "$CONFIG"; then
  echo "FATAL: configuration invalid; refusing to submit" >&2
  exit 3
fi

PREREG_HASH="$("$PY" -c "
from ei.config import ExperimentConfig
print(ExperimentConfig.load('$CONFIG').preregistration_hash())
")"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${PREREG_HASH:0:8}"
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR"

# -------------------------------------------------------------------------------------
# Provenance: recorded BEFORE the run, so a dirty tree is visible in the artifact rather
# than discovered afterwards.
# -------------------------------------------------------------------------------------
{
  echo "run_id=$RUN_ID"
  echo "prereg_hash=$PREREG_HASH"
  echo "config=$CONFIG"
  echo "utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "python=$($PY -V 2>&1)"
  if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_branch=$(git rev-parse --abbrev-ref HEAD)"
    if [[ -n "$(git status --porcelain)" ]]; then
      echo "git_dirty=true"
      echo "WARNING: working tree is dirty; this run is NOT reproducible from the commit" >&2
      git status --porcelain > "$RUN_DIR/git_dirty_files.txt"
      git diff > "$RUN_DIR/uncommitted.patch" || true
    else
      echo "git_dirty=false"
    fi
  fi
  command -v nvidia-smi >/dev/null 2>&1 && \
    echo "gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd';' -)" || \
    echo "gpus=none"
} > "$RUN_DIR/provenance.txt"
cp "$CONFIG" "$RUN_DIR/config.snapshot.yaml"

# -------------------------------------------------------------------------------------
# Detection
# -------------------------------------------------------------------------------------
detect_scheduler() {
  [[ -n "$FORCE" ]] && { echo "$FORCE"; return; }
  [[ -n "${SCHEDULER:-}" ]] && { echo "$SCHEDULER"; return; }
  if command -v sbatch >/dev/null 2>&1 && scontrol ping >/dev/null 2>&1; then
    echo "slurm"; return
  fi
  if [[ -n "${KUBERNETES_SERVICE_HOST:-}" ]] || \
     { command -v kubectl >/dev/null 2>&1 && kubectl cluster-info >/dev/null 2>&1; }; then
    echo "k8s"; return
  fi
  if [[ -n "${RAY_ADDRESS:-}" ]] || \
     { command -v ray >/dev/null 2>&1 && ray status >/dev/null 2>&1; }; then
    echo "ray"; return
  fi
  echo "local"
}

SCHED="$(detect_scheduler)"
NPROC="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
echo "==> scheduler : $SCHED"
echo "==> run dir   : $RUN_DIR"
echo "==> prereg    : $PREREG_HASH"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run: nothing submitted)"
  exit 0
fi

CMD=("$PY" -m ei.cli run "$CONFIG" --out "$RUN_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

case "$SCHED" in
  slurm)
    SB="$RUN_DIR/submit.sbatch"
    cat > "$SB" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ei-${PREREG_HASH:0:8}
#SBATCH --output=$RUN_DIR/slurm-%j.out
#SBATCH --error=$RUN_DIR/slurm-%j.err
#SBATCH --time=${EI_TIME:-04:00:00}
#SBATCH --cpus-per-task=${EI_CPUS:-8}
#SBATCH --mem=${EI_MEM:-64G}
${EI_GPUS:+#SBATCH --gres=gpu:$EI_GPUS}
${EI_PARTITION:+#SBATCH --partition=$EI_PARTITION}
set -Eeuo pipefail
cd "$REPO_ROOT"
# Determinism: thread counts change floating-point reduction order, which changes
# eigenvalues in the last digits. Pinned so reruns are bit-comparable.
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=\$OMP_NUM_THREADS
export PYTHONHASHSEED=0
srun ${CMD[*]}
EOF
    echo "==> sbatch $SB"
    sbatch "$SB"
    ;;

  k8s)
    JOB="ei-${PREREG_HASH:0:8}-$(date +%s)"
    MANIFEST="$RUN_DIR/job.yaml"
    cat > "$MANIFEST" <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB
  labels: { app: ei, prereg: "${PREREG_HASH:0:8}" }
spec:
  backoffLimit: 0          # a failed statistical run must not silently retry
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: ei
          image: ${EI_IMAGE:-ei:0.1.0}
          imagePullPolicy: IfNotPresent
          command: ["python", "-m", "ei.cli", "run", "$CONFIG", "--out", "/work/$RUN_DIR"]
          env:
            - { name: OMP_NUM_THREADS, value: "${EI_CPUS:-8}" }
            - { name: PYTHONHASHSEED,  value: "0" }
          resources:
            requests: { cpu: "${EI_CPUS:-8}", memory: "${EI_MEM:-64Gi}" }
            limits:
              cpu: "${EI_CPUS:-8}"
              memory: "${EI_MEM:-64Gi}"
              ${EI_GPUS:+nvidia.com/gpu: "$EI_GPUS"}
          volumeMounts: [{ name: work, mountPath: /work }]
      volumes:
        - name: work
          persistentVolumeClaim: { claimName: ${EI_PVC:-ei-results} }
EOF
    echo "==> kubectl apply -f $MANIFEST"
    kubectl apply -f "$MANIFEST"
    ;;

  ray)
    echo "==> ray submit"
    RAY_ADDRESS="${RAY_ADDRESS:-auto}" "$PY" - <<PYEOF
import ray, subprocess, sys
ray.init(address="${RAY_ADDRESS:-auto}", ignore_reinit_error=True)

@ray.remote(num_cpus=int("${EI_CPUS:-8}"), num_gpus=int("${EI_GPUS:-0}"))
def run():
    return subprocess.call(${CMD[@]@Q} if False else ["$PY", "-m", "ei.cli", "run",
                           "$CONFIG", "--out", "$RUN_DIR"])

sys.exit(ray.get(run.remote()))
PYEOF
    ;;

  local)
    export OMP_NUM_THREADS="${EI_CPUS:-$NPROC}"
    export MKL_NUM_THREADS="$OMP_NUM_THREADS"
    export PYTHONHASHSEED=0
    echo "==> running locally with $OMP_NUM_THREADS threads"
    "${CMD[@]}" 2>&1 | tee "$RUN_DIR/run.log"
    ;;

  *)
    echo "FATAL: unknown scheduler '$SCHED'" >&2
    exit 4
    ;;
esac

echo "==> artifacts: $RUN_DIR"
