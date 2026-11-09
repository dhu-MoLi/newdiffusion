from __future__ import annotations

import json

import torch


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert torch.cuda.device_count() == 1, "P0 expects one visible GPU"
    props = torch.cuda.get_device_properties(0)
    x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()
    report = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": props.name,
        "memory_bytes": props.total_memory,
        "finite": bool(torch.isfinite(y).all().item()),
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    assert report["gpu"] == "NVIDIA GeForce RTX 3090"
    assert report["memory_bytes"] >= 24_000_000_000
    assert report["finite"]


if __name__ == "__main__":
    main()
