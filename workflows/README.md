# CIQ Fuzzball workflow status

No Fuzzball YAML is included or claimed valid. Team 12's actual Fuzzball deployment is v4.3.1; verify correct syntax, credentials handling, authorized resource definition, storage volume access, NCCL network, ARM64/CUDA image and GPU visibility from a **successful official sample workflow** after access returns Sunday at 9 AM CT.

Then implement smoke test -> one node inference -> two-node independent partitioned inference -> one/two node distributed training (stretch). Keep real GPU logs in private artifacts, never invent timings. See `docs/IMPLEMENTATION_RUNBOOK.md`.

## Inference API / dashboard templates (feat/api-gpu-deploy)

`00_gpu_smoke.yaml.tmpl`, `06_inference_batch.yaml.tmpl` and `07_serve.yaml.tmpl` are **unvalidated templates**
based on the public Fuzzball 4.4 syntax guide. Render them with `scripts/fuzzball/render_workflow.py` and follow
the gates in `docs/DEPLOYMENT.md` §4 on Team 12 v4.3.1 before treating any of them as working.
