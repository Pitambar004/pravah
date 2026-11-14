# ARM64 + GB10 image

The CPU-first scaffold intentionally does not prescribe a fictional CUDA image. Ask CIQ for a verified GB10-compatible ARM64 container and appropriate PyTorch CUDA stack; record its digest. Test `uname -m`, `nvidia-smi`, and `torch.cuda.is_available()` inside an authorized Team 12 container before calling any run GPU-backed. Do not assume direct SSH/host access. Refer to runbook hardware and Fuzzball gates.
