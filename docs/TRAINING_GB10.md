# Training on Fuzzball: 1 GB10 node vs 2 GB10 nodes

This page covers two experiments: training on one GB10 node, then the same training on two GB10 nodes. Both
templates are **UNVERIFIED on Team 12's Fuzzball v4.3.1**. The two-node template follows CIQ's 4.4
`generic`/NCCL docs, so confirm its syntax with a CIQ mentor before you rely on it.

## Before you start

1. Pass Gates A–C in the runbook (§10, P4-04): network, both Team 12 nodes, and the CUDA smoke test in the
   **same image**.
2. Rebuild the API image. It now contains `scripts/train_ddp.py`, see `infra/docker/Dockerfile.api`.
3. Stage the real pairs manifest at `/data/manifests/s2_pairs.json`, then dry-run it:
   `python scripts/train_ddp.py --pairs ... --output-dir /tmp/x --dry-run`.

## Render both experiments with the same workload

```bash
IMG=docker://<registry>/pravah-api:<tag>
python scripts/fuzzball/render_workflow.py workflows/fuzzball/05_train_1node.yaml.tmpl \
  --set API_IMAGE=$IMG --set RUN_ID=gb10_1node_a --set GLOBAL_BATCH=16 --set EPOCHS=3
python scripts/fuzzball/render_workflow.py workflows/fuzzball/05_train_2node_ddp.yaml.tmpl \
  --set API_IMAGE=$IMG --set RUN_ID=gb10_2node_a --set GLOBAL_BATCH=16 --set EPOCHS=3
```

Both runs must use the same `GLOBAL_BATCH`, `EPOCHS`, `MAX_STEPS`, `PAIRS` and seed, and `GLOBAL_BATCH` must be
even. Each rank then trains on `GLOBAL_BATCH / WORLD_SIZE` samples per step, so both runs make the same number of
optimizer steps on the same data. Validate the rendered files with Fuzzball before you submit them. Use a new
`RUN_ID` for every attempt, because the script refuses to append to an old run folder.

## Watch it live

Point the dashboard at the training folder:

```bash
PRAVAH_TRAINING_DIR=/data/outputs/training streamlit run app/Home.py
```

The **Training** page refreshes every 10 s with:

- loss curves by step and by measured training time, with 1-node and 2-node runs overlaid;
- workload-equivalence checks and the measured speedup, shown only when every check passes;
- a downloadable `training_benchmark.json`.

## What counts as proof of two-node training

- `training_summary.json` contains `world_size: 2`, `distinct_hostnames: 2` and `cuda_verified: true`.
- The Fuzzball job logs show two distinct Team 12 node IDs. Hostnames inside a container are not proof of
  separate physical nodes.
- The NCCL init lines (`NCCL_DEBUG=INFO`) show both ranks.

If the two-node run is slower, for example because of NCCL or network overhead on a small model, report that
result as measured.
