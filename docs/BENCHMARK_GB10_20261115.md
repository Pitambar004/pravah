# GB10 training benchmark: 1 node vs 2 nodes (15 Nov 2026)

These numbers were measured on Team 12 Fuzzball v4.3.1 (`fb-hack`). One run of each configuration; no
repeats yet, so run-to-run variance is unknown.

| | A: 1 node | B: 2 nodes (PyTorch DDP, NCCL) |
|---|---|---|
| Fuzzball workflow | `101d4e6a-ddda-4694-8793-f1e32fbc2881` | `49e93d38-8928-4a99-af24-9e0f4995e3d4` |
| Physical hosts (Fuzzball stage record) | `192.168.99.33` | `192.168.99.33`, `192.168.99.32` |
| GPU per rank | NVIDIA GB10, `cuda:0`, CUDA verified | NVIDIA GB10 on both ranks, CUDA verified |
| Optimizer steps / global batch | 500 / 8 | 500 / 8 |
| Training time (data loading + compute) | 486.1 s | 268.8 s |
| Throughput | 8.23 samples/s | 14.88 samples/s |
| **Speed-up / scaling efficiency** | — | **1.81× / 90 %** |

## What was identical

All nine checks on the dashboard Training page passed:
- training data: manifest `quick_sen1_with_weak.json`, SHA-256 `641aa43d…`, 4,409 train / 86 val / 88 test
- global batch, step count, seed (42) and learning rate (0.001)
- world size 1 vs 2, two distinct hosts, CUDA on every rank

Both runs used the stock image `nvcr.io/nvidia/pytorch:25.10-py3` and PRAVAH code bundles built with
`scripts/fuzzball/prepare_world_bundle.py`. A used bundle `6cbd4398…` (commit `8fa2495`). B used `22b19f69…`
(commit `f0ae676`), which differs only by progress-log printing every 20 steps.

## How to read it

- **Model:** PRAVAH's own small MiniUNet (about 0.49 M parameters), trained from scratch. This is not Prithvi
  or Dhara.
- **Likely bottleneck: data loading.** Throughput is low for a model this small, and the data loader ran with
  `num_workers=0` while reading from the shared volume. Part of the 2-node speed-up may come from two nodes
  reading data in parallel rather than from GPU compute. This has not been tested.
- **Shared load:** the world-data download job (`014ead45…`) was running on the cluster during both runs.
- **Not a quality comparison.** Validation IoU was 0.694 for A and 0.729 for B, on 86 hand-labelled chips. Both
  runs stopped at 500 steps, less than one epoch, and DDP changes the sample order. Use the hand-labelled test
  split and the Bolivia holdout for model quality.

Run folders: `outputs/training/bench_{1,2}node_q1/` (not in Git), fetched from
`fb://group/pravah/bench/q1/`.
