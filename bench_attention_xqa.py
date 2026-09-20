#!/usr/bin/env python3
"""XQA entry point; keep bench_attention.py in the same directory.

Matches the public direct API selected by vLLM0.28 on SM121. Default pages=16.
Example:
  python bench_attention_xqa.py --batches 1 --query-lengths 1 4 --context 256
  python bench_attention_xqa.py --output attention_xqa_results.json

Pinned source:
 https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16/flashinfer/decode.py#L3524
 https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/flashinfer.py#L2391
"""

import sys

from bench_attention import main


if __name__ == "__main__":
    if any(arg == "--kernel" or arg.startswith("--kernel=") for arg in sys.argv[1:]):
        raise SystemExit("This entry point fixes --kernel xqa; use bench_attention.py for other kernels.")
    sys.argv[1:1] = ["--kernel", "xqa"]
    raise SystemExit(main())
