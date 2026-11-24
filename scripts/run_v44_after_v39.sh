#!/usr/bin/env bash
set -euo pipefail

project=/home/dzliu/pzliu/sap-rl
campaign="${project}/runs/turtle-v44-end-freeze-ablation"
source_root="${campaign}/source"
parent_run="${project}/runs/turtle-league-v39-unrestricted-from-v38-250"
parent=/workspace/runs/turtle-league-v39-unrestricted-from-v38-250/update-002250.pt
run=/workspace/runs/turtle-v44-end-freeze-ablation
image=sha256:3ed8782e02a5d46e400bc39a5a77654c26e8a86b1d210c160e5965b765955ce1
preflight="${project}/scripts/v44_preflight.py"

[[ -f "${source_root}/sapai-gym/sapai_gym1/rl/end_freeze.py" ]]
python3 "${preflight}" source "${source_root}" "${campaign}/source-manifest.json"
echo 'v44 waiting for the complete v39-2250 evaluation and natural service completion'
while true; do
  ready=0
  python3 "${preflight}" parent "${parent_run}" || ready=$?
  if (( ready != 0 && ready != 3 )); then exit "${ready}"; fi
  service_state=$(systemctl --user show sap-v39-unrestricted.service -p ActiveState --value) || service_state=unknown
  service_result=$(systemctl --user show sap-v39-unrestricted.service -p Result --value) || service_result=unknown
  if (( ready == 0 )) && [[ "${service_state}" == inactive && "${service_result}" == success ]]; then break; fi
  sleep 30
done

# Preserve the exact common league after v39 stops writing it. Never rewrite
# a branch's evolving league on resume, and never promote a smoke checkpoint.
if [[ ! -f "${campaign}/common-league.json" ]]; then
  cp "${parent_run}/league.json" "${campaign}/common-league.json"
  sha256sum "${parent_run}/update-002250.pt" "${campaign}/common-league.json" > "${campaign}/common-parent.sha256"
fi

wait_for_gpu() {
  while true; do
    if gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
      if [[ -z "${gpu_pids//[[:space:]]/}" ]]; then return; fi
    fi
    # Never stop unrelated interpretation/vLLM work to satisfy this experiment.
    sleep 30
  done
}

run_docker() {
  local name=$1
  shift
  wait_for_gpu
  python3 "${preflight}" source "${source_root}" "${campaign}/source-manifest.json"
  if docker inspect "${name}" >/dev/null 2>&1; then
    if [[ $(docker inspect -f '{{.State.Running}}' "${name}") == true ]]; then
      echo "Refusing to replace running ${name}" >&2
      return 1
    fi
    docker rm "${name}" >/dev/null
  fi
  docker run --name "${name}" --gpus all --ipc host --network none \
    -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
    -v "${source_root}/sapai:/workspace/sapai:ro" \
    -v "${source_root}/sapai-gym:/workspace/sapai-gym:ro" \
    -v "${project}/runs:/workspace/runs" -w /workspace/sapai-gym \
    "${image}" "$@"
}

train_branch() {
  local branch=$1
  local target=$2
  local seed=$3
  local joint=$4
  local directory="${campaign}/${branch}"
  local resume="${parent}"
  local previous=2250
  local latest digits
  # Re-read recovery state AFTER the device becomes free. A detached prior
  # invocation may still be completing a later checkpoint while we wait.
  wait_for_gpu
  mkdir -p "${directory}"
  latest=$(find "${directory}" -maxdepth 1 -type f -name 'update-*.pt' -printf '%f\n' | sort -V | tail -1)
  if [[ -n "${latest}" ]]; then
    digits="${latest#update-}"; digits="${digits%.pt}"
    previous=$((10#${digits}))
    resume="${run}/${branch}/${latest}"
  elif [[ ! -f "${directory}/league.json" ]]; then
    cp "${campaign}/common-league.json" "${directory}/league.json"
  fi
  if (( previous >= target )); then
    if ! grep -q "^finished updates=${target} " "${directory}/training.log" 2>/dev/null; then
      if [[ $(docker inspect -f '{{.State.Status}}:{{.State.ExitCode}}' "sap-v44-${branch}" 2>/dev/null) == exited:0 ]]; then
        docker logs "sap-v44-${branch}" >> "${directory}/training.log" 2>&1
      fi
    fi
    return
  fi
  local extra=()
  if [[ "${joint}" == yes ]]; then
    extra=(--end-freeze --end-freeze-learning-rate 5e-5)
  fi
  # Identical actor/learner settings and initial league for both arms. Training
  # is fixed at 1,000 additional updates; actual elapsed compute is reported,
  # not mislabeled as exactly equal GPU-hours.
  run_docker "sap-v44-${branch}" python scripts/train_entity_selfplay.py \
    --distributed --device cuda --updates "${target}" \
    --resume-checkpoint "${resume}" --resume-learning-rate 2e-6 \
    --num-envs 32 --games-per-actor 3 --segments-per-update 64 \
    --transitions-per-update 8192 --trajectory-queue-size 1024 \
    --inference-batch-delay-ms 5 --max-behavior-policy-lag 1 \
    --checkpoint-every 100 --checkpoint-dir "${run}/${branch}" \
    --discount-mode per_turn --discount 0.99 --battle-aux-cost 0.05 \
    --opponent-aux-cost 0.1 --opponent-module-learning-rate 1e-4 \
    --opponent-adapter-learning-rate 1e-5 --max-historical-residents 4 \
    --allow-simulator-mismatch --seed "$((seed + previous - 2250))" \
    "${extra[@]}" 2>&1 | tee -a "${directory}/training.log"
}

train_branch smoke-control 2253 4400101 no
python3 "${preflight}" smoke "${campaign}/smoke-control/training.log" --update 2253
train_branch smoke-end-freeze 2253 4400101 yes
python3 "${preflight}" smoke "${campaign}/smoke-end-freeze/training.log" --update 2253 --joint
python3 "${preflight}" compare-smokes "${campaign}/smoke-control/training.log" "${campaign}/smoke-end-freeze/training.log"

# Both full arms restart from the common parent, not the smoke outputs.
train_branch control 3250 4400201 no
train_branch end-freeze 3250 4400201 yes

mkdir -p "${campaign}/evaluations"
evaluate() {
  local candidate=$1 opponent=$2 label=$3 seed=$4
  if [[ -f "${campaign}/evaluations/${label}.json" ]]; then return; fi
  run_docker sap-v44-evaluation python scripts/evaluate_paired_policy.py \
    "${candidate}" --opponent checkpoint --opponent-checkpoint "${opponent}" \
    --device cuda --batch-size 64 --pairs 100 --seed "${seed}" \
    --progress-every 100 --output "${run}/evaluations/${label}.json" \
    2>&1 | tee -a "${campaign}/evaluations/${label}.log"
}

evaluate "${run}/end-freeze/update-003250.pt" "${run}/control/update-003250.pt" head-vs-control 4400603
for branch in control end-freeze; do
  evaluate "${run}/${branch}/update-003250.pt" "${parent}" "${branch}-vs-parent" 4400601
  evaluate "${run}/${branch}/update-003250.pt" \
    /workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt \
    "${branch}-vs-v31" 4400602
done
echo 'v44 finished both full training windows and five paired evaluations; no automatic promotion or rollback'
