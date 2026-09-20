# Third-party references

- `vllm028/`: selected source from [vLLM v0.28.0](https://github.com/vllm-project/vllm/tree/v0.28.0), [Apache 2.0 license](licenses/vllm-Apache-2.0.txt). Embedded Flash Linear Attention code retains its copyright notices and [MIT license](licenses/flash-linear-attention-MIT.txt).
- `oscar-hybrid-source/`: selected research code from [OSCAR commit 19f85e13059de3da60686af4cbdd778b5671d9ff](https://github.com/FutureMLS-Lab/OSCAR/tree/19f85e13059de3da60686af4cbdd778b5671d9ff), [MIT license](licenses/OSCAR-MIT.txt). The selected recurrent kernel derives from Flash Linear Attention and retains its authors' notices.
- `serving_recipe_sources/parallel_draft/dflash-readme.md`: captured [DFlash model card](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash), declared Apache 2.0; see the Apache license above. This candidate was not benchmarked on the workstation.
- The complete upstream checkouts, virtual environments, generated caches, and downloaded model weights are not versioned here. Pinned model revisions and upstream links identify the exact tested runtime and referenced candidates.
- This repository does not assign a new license to third-party code or model weights.
