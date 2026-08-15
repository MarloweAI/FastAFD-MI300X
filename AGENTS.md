# Coding-agent instructions: reproduce FastAFD on MI300X

Scope: recreate and run this repository on a Linux x86-64 AMD MI300X (`gfx942`)
machine without changing the serving implementation.

1. Verify `/opt/rocm/.info/version` is ROCm 7.x, `hipcc` works, `rocminfo`
   reports `gfx942`, and the current user can access `/dev/kfd` and `/dev/dri`.
2. Verify `conda` or Miniforge is installed. Run `./bootstrap_rocm.sh`. If conda
   is elsewhere, set `CONDA_EXE`; if ROCm is elsewhere, set `ROCM_PATH`.
3. Export `ENV_PREFIX="$PWD/.conda-env"`. Run
   `"$ENV_PREFIX/bin/python" scripts/check_rocm_runtime.py` and stop on failure.
4. Download weights with
   `"$ENV_PREFIX/bin/hf" download openai/gpt-oss-120b --local-dir "$PWD/models/gpt-oss-120b"`.
   Export `MODEL="$PWD/models/gpt-oss-120b"`.
5. For the validated configuration, export `MINISGL_MXFP4_PACKED=1` and run
   `TP=4 GPUS=0,1,2,3 GRAPH_MAX_BS=32 ./run_col_rocm.sh`.
6. Wait for `/v1/chat/completions` on port 19295, then run `./ask_rocm.sh`.
7. Run `experiments/run_decode_grid.sh` only after the first request has
   completed and JIT/autotuning has settled.

Do not install NVIDIA NCCL, CUDA `triton`, FlashInfer, DeepEP, or DeepGEMM.
Use the ROCm torch wheel and `pytorch-triton-rocm` from the supplied environment
files. Never set both `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`.

Do not enable packed MXFP4 on `run_afd_rocm.sh`: that AFD branch is implemented
but was not validated live. Use `NUM_MB=1`, `GRAPH_MAX_BS=0`, and make `MLP_EP`
match `MLP_TP` for the documented gpt-oss layouts.

Expected validated versions are recorded in README.md and
environment.rocm7.pinned.yml. Preserve those versions before investigating any
runtime issue; upgrading dependencies changes the experiment.
