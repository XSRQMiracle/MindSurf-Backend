"""Benchmark native TransformerLM inference against vLLM.

The native backend loads the training checkpoint and calls the same
TransformerLM.generate() method used by scripts/inference.py. The vLLM backend
loads the converted Hugging Face directory. Backends run in separate processes
so that CUDA allocations from one implementation cannot affect the other.

Example:
    uv run scripts/benchmark_inference.py \
        --batch-sizes 1,8,32 \
        --max-new-tokens 128 \
        --repeats 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from transformers import AutoConfig, AutoTokenizer

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

Backend = Literal["pytorch", "vllm"]

DEFAULT_CHECKPOINT = Path("models/checkpoints/test/final_model.pt")
DEFAULT_VLLM_MODEL = Path("models/transformers/mindsurf2llama")
DEFAULT_MODEL_CONFIG = Path("configs/model/minimind_small.yaml")
DEFAULT_TOKENIZER = Path("models/tokenizers/minimind_tokenizer")

DEFAULT_PROMPTS = (
    "人工智能的发展将会",
    "请简要介绍机器学习的基本概念。",
    "深度学习模型训练时需要注意",
    "写一个关于未来城市的短故事：",
    "自然语言处理的主要任务包括",
    "Explain the difference between training and inference.",
    "Once upon a time, there was a small language model",
    "The most important property of a reliable system is",
)

DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Measured throughput for one backend and batch size."""

    backend: Backend
    batch_size: int
    repeats: int
    elapsed_seconds: float
    prompt_tokens: int
    output_tokens: int

    @property
    def average_batch_latency_ms(self) -> float:
        return self.elapsed_seconds * 1000 / self.repeats

    @property
    def requests_per_second(self) -> float:
        return self.batch_size * self.repeats / self.elapsed_seconds

    @property
    def output_tokens_per_second(self) -> float:
        return self.output_tokens / self.elapsed_seconds


@dataclass(frozen=True)
class BackendReport:
    """Load time and measurements for one inference backend."""

    backend: Backend
    load_seconds: float
    metrics: tuple[BenchmarkMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "load_seconds": self.load_seconds,
            "metrics": [asdict(metric) for metric in self.metrics],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> BackendReport:
        backend = cast(Backend, value["backend"])
        raw_metrics = cast(list[dict[str, Any]], value["metrics"])
        metrics = tuple(BenchmarkMetrics(**metric) for metric in raw_metrics)
        return cls(
            backend=backend,
            load_seconds=float(cast(float, value["load_seconds"])),
            metrics=metrics,
        )


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive batch sizes."""
    try:
        batch_sizes = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc

    if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise argparse.ArgumentTypeError("batch sizes must not contain duplicates")
    return batch_sizes


def load_prompts(path: Path | None) -> tuple[str, ...]:
    """Load non-empty prompts from a UTF-8 text file or use defaults."""
    if path is None:
        return DEFAULT_PROMPTS

    prompts = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    prompts = tuple(prompt for prompt in prompts if prompt)
    if not prompts:
        raise ValueError(f"Prompt file contains no non-empty lines: {path}")
    return prompts


def synchronize_cuda() -> None:
    """Wait for queued CUDA work before taking a timing boundary."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def require_cuda() -> None:
    """Fail with a useful message when the benchmark has no CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires a CUDA GPU. Run it on the Linux GPU host used for vLLM."
        )


def load_model_config(path: Path) -> Any:
    """Load the native ModelConfig through the converter's shared parser."""
    from scripts.convert_model import load_model_config as load_config

    return load_config(path)


def load_checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load the native state dict through the converter's shared parser."""
    from scripts.convert_model import load_checkpoint_state_dict as load_state_dict

    return load_state_dict(path)


def prepare_tokenizer(path: Path, trust_remote_code: bool) -> Any:
    """Load the tokenizer used for both benchmark backends."""
    return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)


def encode_prompts(tokenizer: Any, prompts: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    """Tokenize prompts once, outside timed generation regions."""
    encoded: list[tuple[int, ...]] = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        if not token_ids:
            raise ValueError(f"Tokenizer produced an empty prompt: {prompt!r}")
        encoded.append(tuple(int(token_id) for token_id in token_ids))
    return tuple(encoded)


def validate_model_pair(args: argparse.Namespace) -> None:
    """Verify that the checkpoint config, converted model, and tokenizer match."""
    required_paths = {
        "checkpoint": args.checkpoint,
        "vLLM model": args.vllm_model,
        "model config": args.model_config,
        "tokenizer": args.tokenizer,
    }
    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    native_config = load_model_config(args.model_config)
    converted_config = AutoConfig.from_pretrained(
        args.vllm_model,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = prepare_tokenizer(args.tokenizer, args.trust_remote_code)

    expected_pairs = {
        "vocab_size": (native_config.vocab_size, converted_config.vocab_size),
        "hidden_size": (native_config.n_embed, converted_config.hidden_size),
        "intermediate_size": (native_config.hidden_dim, converted_config.intermediate_size),
        "num_hidden_layers": (native_config.n_layer, converted_config.num_hidden_layers),
        "num_attention_heads": (native_config.n_head, converted_config.num_attention_heads),
        "max_position_embeddings": (
            native_config.max_seq_len,
            converted_config.max_position_embeddings,
        ),
    }
    mismatches = {name: values for name, values in expected_pairs.items() if values[0] != values[1]}
    if mismatches:
        raise ValueError(f"Native and converted model configs do not match: {mismatches}")
    if len(tokenizer) != native_config.vocab_size:
        raise ValueError(
            "Tokenizer size does not match checkpoint config: "
            f"tokenizer={len(tokenizer)}, vocab_size={native_config.vocab_size}"
        )
    if args.max_model_len > native_config.max_seq_len:
        raise ValueError(
            f"--max-model-len cannot exceed checkpoint max_seq_len={native_config.max_seq_len}"
        )


def validate_prompt_lengths(
    encoded_prompts: tuple[tuple[int, ...], ...],
    max_new_tokens: int,
    max_model_len: int,
) -> None:
    """Keep generation inside the trained context window for both backends."""
    longest_prompt = max(len(prompt) for prompt in encoded_prompts)
    if longest_prompt + max_new_tokens > max_model_len:
        raise ValueError(
            "Prompt plus generated tokens exceeds --max-model-len: "
            f"prompt={longest_prompt}, generation={max_new_tokens}, limit={max_model_len}"
        )


def run_pytorch_batch(
    model: Any,
    prompt_token_ids: tuple[int, ...],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[int, int]:
    """Generate through TransformerLM.generate() and return token counts."""
    input_ids = torch.tensor(
        [prompt_token_ids] * batch_size,
        dtype=torch.long,
        device="cuda",
    )
    input_width = int(input_ids.shape[1])
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=None,
        )

    prompt_tokens = batch_size * input_width
    output_tokens = int(output_ids.shape[0]) * (int(output_ids.shape[1]) - input_width)
    return prompt_tokens, output_tokens


def benchmark_pytorch(args: argparse.Namespace, prompts: tuple[str, ...]) -> BackendReport:
    """Measure the same native PyTorch path used by scripts/inference.py."""
    require_cuda()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    load_started = time.perf_counter()
    tokenizer = prepare_tokenizer(args.tokenizer, args.trust_remote_code)
    model_config = load_model_config(args.model_config)
    state_dict = load_checkpoint_state_dict(args.checkpoint)

    from python_starter.core.model import TransformerLM

    model = TransformerLM(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device="cuda", dtype=DTYPES[args.dtype])
    model.eval()
    encoded_prompts = encode_prompts(tokenizer, prompts)
    validate_prompt_lengths(encoded_prompts, args.max_new_tokens, args.max_model_len)
    synchronize_cuda()
    load_seconds = time.perf_counter() - load_started

    metrics: list[BenchmarkMetrics] = []
    for batch_size in args.batch_sizes:
        for warmup_index in range(args.warmup):
            prompt_token_ids = encoded_prompts[warmup_index % len(encoded_prompts)]
            run_pytorch_batch(
                model,
                prompt_token_ids,
                batch_size,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )

        synchronize_cuda()
        started = time.perf_counter()
        prompt_tokens = 0
        output_tokens = 0
        for repeat_index in range(args.repeats):
            prompt_token_ids = encoded_prompts[repeat_index % len(encoded_prompts)]
            current_prompt_tokens, current_output_tokens = run_pytorch_batch(
                model,
                prompt_token_ids,
                batch_size,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
            prompt_tokens += current_prompt_tokens
            output_tokens += current_output_tokens
        synchronize_cuda()
        elapsed_seconds = time.perf_counter() - started

        metrics.append(
            BenchmarkMetrics(
                backend="pytorch",
                batch_size=batch_size,
                repeats=args.repeats,
                elapsed_seconds=elapsed_seconds,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        )

    return BackendReport("pytorch", load_seconds, tuple(metrics))


def make_vllm_token_batch(prompt_token_ids: tuple[int, ...], batch_size: int) -> list[Any]:
    """Build pre-tokenized vLLM inputs matching the native tokenizer output."""
    return [{"prompt_token_ids": list(prompt_token_ids)} for _ in range(batch_size)]


def run_vllm_batch(
    llm: Any,
    sampling_params: Any,
    prompt_token_ids: tuple[int, ...],
    batch_size: int,
) -> tuple[int, int]:
    """Generate one pre-tokenized vLLM batch and return token counts."""
    outputs = llm.generate(
        make_vllm_token_batch(prompt_token_ids, batch_size),
        sampling_params,
        use_tqdm=False,
    )
    prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    output_tokens = sum(
        len(completion.token_ids) for output in outputs for completion in output.outputs
    )
    return prompt_tokens, output_tokens


def benchmark_vllm(args: argparse.Namespace, prompts: tuple[str, ...]) -> BackendReport:
    """Measure batched vLLM offline generation."""
    require_cuda()
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed; run `uv sync --extra vllm` on Linux") from exc

    load_started = time.perf_counter()
    tokenizer = prepare_tokenizer(args.tokenizer, args.trust_remote_code)
    encoded_prompts = encode_prompts(tokenizer, prompts)
    validate_prompt_lengths(encoded_prompts, args.max_new_tokens, args.max_model_len)
    llm = LLM(
        model=str(args.vllm_model),
        tokenizer=str(args.tokenizer),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        disable_log_stats=True,
        seed=args.seed,
    )
    load_seconds = time.perf_counter() - load_started
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
        seed=args.seed,
    )

    metrics: list[BenchmarkMetrics] = []
    for batch_size in args.batch_sizes:
        for warmup_index in range(args.warmup):
            prompt_token_ids = encoded_prompts[warmup_index % len(encoded_prompts)]
            run_vllm_batch(llm, sampling_params, prompt_token_ids, batch_size)

        started = time.perf_counter()
        prompt_tokens = 0
        output_tokens = 0
        for repeat_index in range(args.repeats):
            prompt_token_ids = encoded_prompts[repeat_index % len(encoded_prompts)]
            current_prompt_tokens, current_output_tokens = run_vllm_batch(
                llm,
                sampling_params,
                prompt_token_ids,
                batch_size,
            )
            prompt_tokens += current_prompt_tokens
            output_tokens += current_output_tokens
        elapsed_seconds = time.perf_counter() - started

        metrics.append(
            BenchmarkMetrics(
                backend="vllm",
                batch_size=batch_size,
                repeats=args.repeats,
                elapsed_seconds=elapsed_seconds,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        )

    return BackendReport("vllm", load_seconds, tuple(metrics))


def print_reports(reports: tuple[BackendReport, ...]) -> None:
    """Print backend metrics and vLLM speedups as compact tables."""
    print("\nLoad time")
    print(f"{'Backend':<12} {'Seconds':>10}")
    for report in reports:
        print(f"{report.backend:<12} {report.load_seconds:>10.3f}")

    print("\nGeneration performance (fixed-length sampling)")
    print(
        f"{'Backend':<12} {'Batch':>7} {'Avg batch ms':>14} {'Requests/s':>12} {'Output tok/s':>14}"
    )
    for report in reports:
        for metric in report.metrics:
            print(
                f"{metric.backend:<12} {metric.batch_size:>7} "
                f"{metric.average_batch_latency_ms:>14.2f} "
                f"{metric.requests_per_second:>12.2f} "
                f"{metric.output_tokens_per_second:>14.2f}"
            )

    by_backend = {report.backend: report for report in reports}
    if "pytorch" not in by_backend or "vllm" not in by_backend:
        return

    pytorch_metrics = {metric.batch_size: metric for metric in by_backend["pytorch"].metrics}
    print("\nvLLM speedup by output-token throughput")
    print(f"{'Batch':>7} {'Speedup':>10}")
    for metric in by_backend["vllm"].metrics:
        baseline = pytorch_metrics[metric.batch_size].output_tokens_per_second
        speedup = metric.output_tokens_per_second / baseline
        print(f"{metric.batch_size:>7} {speedup:>9.2f}x")


def write_report(path: Path, report: BackendReport) -> None:
    """Write one subprocess report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def read_report(path: Path) -> BackendReport:
    """Load one subprocess report from JSON."""
    raw_value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise ValueError(f"Invalid benchmark report: {path}")
    return BackendReport.from_dict(cast(dict[str, object], raw_value))


def build_worker_command(
    args: argparse.Namespace,
    backend: Backend,
    report_path: Path,
) -> list[str]:
    """Build a child command containing all benchmark-affecting options."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(args.checkpoint),
        "--vllm-model",
        str(args.vllm_model),
        "--model-config",
        str(args.model_config),
        "--tokenizer",
        str(args.tokenizer),
        "--backend",
        backend,
        "--batch-sizes",
        ",".join(str(size) for size in args.batch_sizes),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--worker-report",
        str(report_path),
    ]
    if args.prompts_file is not None:
        command.extend(("--prompts-file", str(args.prompts_file)))
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def run_isolated_backends(args: argparse.Namespace) -> tuple[BackendReport, ...]:
    """Run each backend in its own process and collect reports."""
    args.report_dir.mkdir(parents=True, exist_ok=True)
    reports: list[BackendReport] = []

    for backend in ("pytorch", "vllm"):
        report_path = args.report_dir / f"{backend}.json"
        print(f"\nRunning {backend} benchmark...")
        subprocess.run(
            build_worker_command(args, backend, report_path),
            check=True,
        )
        reports.append(read_report(report_path))

    return tuple(reports)


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare native TransformerLM and vLLM generation throughput"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vllm-model", type=Path, default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--backend",
        choices=("both", "pytorch", "vllm"),
        default="both",
        help="Backend to benchmark (default: both in isolated processes)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=(1, 8, 32),
        help="Comma-separated batch sizes (default: 1,8,32)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="float32",
        help="Use float32 to match scripts/inference.py; float16 is also supported",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("models/benchmark_results"),
        help="Directory for per-backend JSON reports",
    )
    parser.add_argument("--worker-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_model_len <= 0:
        parser.error("--max-model-len must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args


def main() -> None:
    """Run the selected inference benchmarks."""
    args = parse_args()
    if args.worker_report is None:
        validate_model_pair(args)
    prompts = load_prompts(args.prompts_file)

    if args.backend == "both":
        if args.worker_report is not None:
            raise ValueError("--worker-report cannot be used with --backend both")
        reports = run_isolated_backends(args)
    else:
        backend = cast(Backend, args.backend)
        report = (
            benchmark_pytorch(args, prompts)
            if backend == "pytorch"
            else benchmark_vllm(args, prompts)
        )
        if args.worker_report is not None:
            write_report(args.worker_report, report)
            return
        reports = (report,)

    print_reports(reports)


if __name__ == "__main__":
    main()
