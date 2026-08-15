# Coding-agent instructions: native FastAFD on MI355X

Scope: build and benchmark the best practical MI355X (`gfx950`) implementation.
The MI300X history is a reference, not a compatibility contract. Prefer native
CDNA4/AITER MXFP4 kernels and change the serving implementation when that is the
right MI355X design.

1. Run GPU work through Slurm on one 8-GPU Colovore MI355X node. Record the node,
   image digest, ROCm, PyTorch, AITER, source commit, and exact command in every run.
2. Keep GPT-OSS expert weights packed. The benchmark path must use AITER's gfx950
   MXFP4 MoE kernels and fail loudly if it falls back or dequantizes the weights.
3. Treat an `A:B` split as `A` TP1 attention workers plus `B` TP1 expert workers.
   Use full expert parallelism across the `B` workers. Uneven partitions are valid:
   pad the final local expert shard and mask its nonexistent experts.
4. Exercise all full-node splits `7:1` through `1:7`. Correctness must cover every
   split; performance uses concurrency `4,8,16,32,64,128` on the InferenceX
   8192-input/1024-output random workload.
5. Reproduce the pinned vLLM baseline on the same physical node. Do not compare an
   AFD rerun on one node against only published numbers from another node.
6. Plot total token throughput per physical GPU on x and median interactivity
   output tokens/s (`1 / median TPOT`) on y. Show published vLLM, same-node vLLM,
   every AFD point, per-split frontiers, and the combined AFD Pareto envelope.
7. Never set both `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`. Do not install
   NVIDIA NCCL/CUDA packages into the ROCm environment.

The reproducibility entrypoint is `experiments/mi355x/README.md`; update it when
the benchmark contract or pinned dependencies change.
