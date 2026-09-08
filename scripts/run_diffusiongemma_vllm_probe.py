"""Run a minimal real DiffusionGemma vLLM generation probe.

This script is intentionally separate from the main ARENA test harness because
the current vLLM runtime pins a different torch/CUDA stack than the primary uv
environment. Run it with an isolated vLLM Python executable, not with the main
course interpreter.
"""

from __future__ import annotations

import argparse
import json
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Write one concise sentence explaining why mechanistic interpretability needs controls.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Render the prompt as a single user message with the model tokenizer.",
    )
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="Do not strip special tokens from the decoded output.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)

    rendered_prompt = args.prompt
    if args.use_chat_template:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.model))
        rendered_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    started_at = time.time()
    llm = LLM(
        model=str(args.model),
        runner="generate",
        trust_remote_code=False,
        tensor_parallel_size=1,
        dtype="auto",
        quantization="modelopt_fp4",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        generation_config="vllm",
        hf_overrides={
            "diffusion_sampler": "entropy_bound",
            "diffusion_entropy_bound": 0.1,
        },
        diffusion_config={"canvas_length": args.canvas_length},
    )
    loaded_at = time.time()

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        seed=args.seed,
        skip_special_tokens=not args.include_special_tokens,
    )
    outputs = llm.generate([rendered_prompt], sampling)
    finished_at = time.time()

    result: dict[str, Any] = {
        "model": str(args.model),
        "prompt": args.prompt,
        "rendered_prompt": rendered_prompt,
        "used_chat_template": args.use_chat_template,
        "output": outputs[0].outputs[0].text,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "canvas_length": args.canvas_length,
        "load_seconds": loaded_at - started_at,
        "generate_seconds": finished_at - loaded_at,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": metadata.version("vllm"),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gib": props.total_memory / 1024**3,
                "parent_process_peak_allocated_gib": torch.cuda.max_memory_allocated()
                / 1024**3,
            }
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
