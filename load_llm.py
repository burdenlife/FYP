"""
Batched interface pipeline for ARM LLMs using Hugging Face Transformers
- Uses quantized loading (via bitsandbytes)
- Logs metadata
- Saves JSONL and CSV
"""
import os
import sys
import json
import csv
import time
import math
import hashlib
import argparse
from dataclasses import asdict, dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.utils import is_flash_attn_2_available
from accelerate import init_empty_weights
from tqdm import tqdm

# -----------------------------
# Utilities
# -----------------------------

def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def human_bytes(n: int) -> str:
    if n is None:
        return "NA"
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(suffixes) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {suffixes[i]}"

# -----------------------------
# Schema
# -----------------------------

@dataclass
class GenParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = False
    max_new_tokens: int = 256
    repetition_penalty: float = 1.0
    stop: List[str] = field(default_factory=list)

@dataclass
class RunMeta:
    run_id: str
    created_at: str
    model_id: str
    model_revision: Optional[str]
    precision: str
    quantization: Optional[str]  # "8bit" / "4bit" / None
    device: str
    n_gpus: int
    torch_cuda_version: Optional[str]
    transformers_version: str
    accelerate_version: str
    flash_attn2: bool
    seed: Optional[int]
    batch_size: int
    pad_to_max_length: bool
    max_input_length: int

@dataclass
class Row:
    run_id: str
    uid: str
    index: int
    prompt: str
    prompt_hash: str
    prompt_len_tokens: int
    output: str
    output_len_tokens: int
    input_plus_output_tokens: int
    params: Dict[str, Any]
    latency_s: float
    tok_per_s: float
    gpu_max_mem: Optional[str]
    err: Optional[str] = None

# -----------------------------
# Core
# -----------------------------

def prepare_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Ensure pad token for batching, typical for LLaMA/Mistral:
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def get_dtype(precision: str):
    precision = precision.lower()
    if precision in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if precision in ["fp16", "float16", "half"]:
        return torch.float16
    if precision in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unknown precision: {precision}")

def load_model(model_id: str, precision: str, load_in_8bit: bool, load_in_4bit: bool, device_map: str):
    dtype = get_dtype(precision)

    kw = dict(device_map=device_map)

    if load_in_8bit and load_in_4bit:
        raise ValueError("Choose only one: --load_in_8bit OR --load_in_4bit")

    if load_in_8bit:
        kw.update(dict(load_in_8bit=True))
    elif load_in_4bit:
        kw.update(dict(load_in_4bit=True, bnb_4bit_compute_dtype=dtype))
    else:
        kw.update(dict(torch_dtype=dtype))

    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    return model

def run_batched_generation(
    model,
    tok,
    prompts: List[str],
    params: GenParams,
    batch_size: int,
    pad_to_max_length: bool,
    max_input_length: int,
    seed: Optional[int] = None,
) -> List[Row]:

    device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    rows: List[Row] = []

    # Pre-tokenize all prompts
    encs = tok(
        prompts,
        padding="max_length" if pad_to_max_length else True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    n = len(prompts)

    # Process in mini-batches
    for start in tqdm(range(0, n, batch_size), desc="Batches"):
        end = min(start + batch_size, n)
        sub_input_ids = encs["input_ids"][start:end].to(device)
        sub_attn = encs["attention_mask"][start:end].to(device)

        if device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass

        t0 = time.perf_counter()

        gen_out = model.generate(
            input_ids=sub_input_ids,
            attention_mask=sub_attn,
            max_new_tokens=params.max_new_tokens,
            do_sample=params.do_sample,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repetition_penalty=params.repetition_penalty,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

        dt = time.perf_counter() - t0

        # Extract only the generated tail for each item
        for i in range(end - start):
            full_ids = gen_out[i].tolist()
            in_len = int(sub_attn[i].sum().item())
            out_ids = full_ids[in_len:]
            out_text = tok.decode(out_ids, skip_special_tokens=True)

            prompt_text = prompts[start + i]
            prompt_hash = sha256_of(prompt_text.strip())
            uid = sha256_of(
                json.dumps(
                    dict(prompt_hash=prompt_hash, params=asdict(params), model_id=str(model.config._name_or_path)),
                    sort_keys=True,
                )
            )

            output_len_tokens = len(out_ids)
            total_tok = in_len + output_len_tokens

            gpu_max_mem = None
            if device.type == "cuda":
                try:
                    peak = torch.cuda.max_memory_allocated(device)
                    gpu_max_mem = human_bytes(peak)
                except Exception:
                    gpu_max_mem = None

            rows.append(
                Row(
                    run_id="",  # filled by caller
                    uid=uid,
                    index=start + i,
                    prompt=prompt_text,
                    prompt_hash=prompt_hash,
                    prompt_len_tokens=in_len,
                    output=out_text,
                    output_len_tokens=output_len_tokens,
                    input_plus_output_tokens=total_tok,
                    params=asdict(params),
                    latency_s=dt,
                    tok_per_s=(output_len_tokens / dt) if dt > 0 else float("nan"),
                    gpu_max_mem=gpu_max_mem,
                    err=None,
                )
            )

    return rows

def save_outputs(rows: List[Row], out_jsonl: Path, out_csv: Path, run_meta: RunMeta):
    # attach run_id to rows and write JSONL
    with out_jsonl.open("a", encoding="utf-8") as f:
        for r in rows:
            r.run_id = run_meta.run_id
            f.write(json.dumps({**asdict(r), "run_meta": asdict(run_meta)}, ensure_ascii=False) + "\n")

    # also write CSV (one row per sample, meta collapsed)
    fieldnames = [
        "run_id","uid","index","prompt_hash","prompt_len_tokens",
        "output_len_tokens","input_plus_output_tokens","latency_s","tok_per_s","gpu_max_mem",
        "model_id","precision","quantization","batch_size","max_input_length","created_at"
    ]
    # deduplicate uids for CSV (latest wins)
    by_uid = {}
    for r in rows:
        by_uid[r.uid] = r
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        for r in by_uid.values():
            writer.writerow({
                "run_id": run_meta.run_id,
                "uid": r.uid,
                "index": r.index,
                "prompt_hash": r.prompt_hash,
                "prompt_len_tokens": r.prompt_len_tokens,
                "output_len_tokens": r.output_len_tokens,
                "input_plus_output_tokens": r.input_plus_output_tokens,
                "latency_s": f"{r.latency_s:.4f}",
                "tok_per_s": f"{r.tok_per_s:.3f}" if r.tok_per_s==r.tok_per_s else "NaN",
                "gpu_max_mem": r.gpu_max_mem or "NA",
                "model_id": run_meta.model_id,
                "precision": run_meta.precision,
                "quantization": run_meta.quantization or "None",
                "batch_size": run_meta.batch_size,
                "max_input_length": run_meta.max_input_length,
                "created_at": run_meta.created_at,
            })

def main():
    p = argparse.ArgumentParser(description="Reusable batched inference for open-source AR LLMs")
    p.add_argument("--model", required=True, help="HF model id, e.g., mistralai/Mistral-7B-Instruct-v0.2 or meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--revision", default=None, help="Specific commit or tag for reproducibility")
    p.add_argument("--precision", default="bf16", choices=["bf16","fp16","fp32"], help="Compute dtype when NOT quantized")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_input_length", type=int, default=2048)
    p.add_argument("--pad_to_max_length", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--prompts_file", default=None, help="Path to a text file with one prompt per line")
    p.add_argument("--out_prefix", default="runs/baseline", help="Output prefix (dir/prefix)")
    args = p.parse_args()

    out_dir = Path(args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = Path(f"{args.out_prefix}.jsonl")
    out_csv = Path(f"{args.out_prefix}.csv")

    # Prompts
    if args.prompts_file is None:
        # Default: 10 benign prompts for acceptance test
        prompts = [
            "List three uses of a paperclip.",
            "Summarize the concept of photosynthesis in one sentence.",
            "Translate to French: 'The cat sits on the mat.'",
            "What is the capital of Japan?",
            "Give me a fun fact about dolphins.",
            "Explain what a hash function is in simple terms.",
            "Write a haiku about rain.",
            "What's the difference between RAM and VRAM?",
            "Provide two safe cooking tips for beginners.",
            "Define entropy in thermodynamics in one sentence."
        ]
    else:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]

    # Resolve model + tokenizer
    model_id = args.model
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tok = prepare_tokenizer(model_id)

    # Load model (quantized or not)
    model = load_model(
        model_id=model_id,
        precision=args.precision,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
    )

    params = GenParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        stop=[],
    )

    quant = "8bit" if args.load_in_8bit else ("4bit" if args.load_in_4bit else None)

    run_meta = RunMeta(
        run_id=sha256_of(now_iso() + model_id + json.dumps(asdict(params))),
        created_at=now_iso(),
        model_id=model_id,
        model_revision=args.revision,
        precision=args.precision,
        quantization=quant,
        device=str(getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))),
        n_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        torch_cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
        transformers_version=__import__("transformers").__version__,
        accelerate_version=__import__("accelerate").__version__,
        flash_attn2=is_flash_attn_2_available(),
        seed=args.seed,
        batch_size=args.batch_size,
        pad_to_max_length=args.pad_to_max_length,
        max_input_length=args.max_input_length,
    )

    rows = run_batched_generation(
        model=model,
        tok=tok,
        prompts=prompts,
        params=params,
        batch_size=args.batch_size,
        pad_to_max_length=args.pad_to_max_length,
        max_input_length=args.max_input_length,
        seed=args.seed,
    )

    save_outputs(rows, out_jsonl=out_jsonl, out_csv=out_csv, run_meta=run_meta)

    ok = len(rows) >= 10 and all((r.err is None and isinstance(r.output, str) and len(r.output) > 0) for r in rows[:10])
    print(f"[ACCEPTANCE] >=10 prompts completed & saved: {'PASS' if ok else 'FAIL'}")
    print(f"Wrote: {out_jsonl} and {out_csv}")

if __name__ == "__main__":
    main()
