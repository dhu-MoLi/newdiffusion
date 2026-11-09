"""Official Hazelnut generate() wiring. Not Module C downstream."""

__all__ = ["GenerateResult", "generate"]


def __getattr__(name: str):
    if name in {"GenerateResult", "generate"}:
        from .generate import GenerateResult, generate

        exports = {
            "GenerateResult": GenerateResult,
            "generate": generate,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
