from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from cadiff.config import load_config
from cadiff.data.mvtec_manifest import build_mvtec_manifest, write_manifest
from cadiff.diagnostics.config import (
    load_diagnostic_config,
    load_noise_replay_config,
)
from cadiff.diagnostics.gate_a_diagnostic import DiagnosticContext, run_diagnostic_phase
from cadiff.diffusion.components import load_frozen_components
from cadiff.p0 import load_experiment_config, load_gate_config, run_p0
from cadiff.p1_v1.config import load_smoke0_config
from cadiff.p1_v1.smoke0 import Smoke0Context, run_smoke0
from cadiff.p2_a0.config import load_p2_a0_config
from cadiff.p2_a0.fewshot import FewshotContext, run_fewshot
from cadiff.p2_a0.fewshot_config import load_p2_a0_fewshot_config
from cadiff.p2_a0.probe import CapacityProbeContext, run_capacity_probe
from cadiff.p2_a0.single_type_config import load_p2_a0_single_type_config
from cadiff.p2_a1.config import load_p2_a1_config
from cadiff.p2_a1.runner import P2A1Context, run_p2_a1
from cadiff.p2_b1.config import load_p2_b1_config
from cadiff.p2_b2.config import load_p2_b2_config, load_p2_b2_experiment_config
from cadiff.p2_b2.runner import P2B2Context, run_p2_b2
from cadiff.p2_b3.config import load_p2_b3_config
from cadiff.p2_b3.runner import P2B3Context, run_p2_b3
from cadiff.p2_c0.config import load_p2_c0_config
from cadiff.p2_c0.runner import P2C0Context, run_p2_c0
from cadiff.p3_a0.config import load_p3_a0_config
from cadiff.p3_a0.runner import P3A0Context, run_p3_a0
from cadiff.v06_b.a2_to_h import run_mask_first_a2_to_h_training
from cadiff.v06_b.a_to_h import run_mask_first_a_to_h_training
from cadiff.v06_b.config import V06BMaskFirstConfig, load_v06_b_config
from cadiff.v06_b.mask_first_panel import run_mask_first_panel
from cadiff.v06_b.mask_first_runner import (
    MaskFirstV06BContext,
    load_mask_first_source_prompts,
    run_mask_first_renderer_continue,
    run_mask_first_training,
)


def _build_manifest(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    records = build_mvtec_manifest(config.dataset_root, args.split_id)
    digest = write_manifest(records, Path(args.output))
    roles = Counter(role for record in records for role in record.roles)
    categories = Counter(record.category for record in records)
    category_defect_types = Counter(f"{record.category}/{record.defect_type}" for record in records)
    report = {
        "sha256": digest,
        "records": len(records),
        "roles": dict(sorted(roles.items())),
        "categories": dict(sorted(categories.items())),
        "category_defect_types": dict(sorted(category_defect_types.items())),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


def _check_model(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    components = load_frozen_components(config.model_root, device, torch.float16)
    torch.cuda.synchronize(device)

    modules = {
        "vae": components.vae,
        "text_encoder": components.text_encoder,
        "unet": components.unet,
    }
    frozen = all(
        not parameter.requires_grad
        for module in modules.values()
        for parameter in module.parameters()
    )
    eval_mode = all(not module.training for module in modules.values())
    component_parameter_dtypes = {
        name: sorted(
            {str(parameter.dtype).removeprefix("torch.") for parameter in module.parameters()}
        )
        for name, module in modules.items()
    }
    if not frozen:
        raise RuntimeError("one or more base parameters are trainable")
    if not eval_mode:
        raise RuntimeError("one or more base modules are not in eval mode")
    if any(dtypes != ["float16"] for dtypes in component_parameter_dtypes.values()):
        raise RuntimeError("one or more base parameters are not float16")

    report = {
        "device": "cuda:0",
        "dtype": "float16",
        "frozen": frozen,
        "eval": eval_mode,
        "component_parameter_dtypes": component_parameter_dtypes,
        "scheduler": components.scheduler.__class__.__name__,
        "max_memory_allocated": torch.cuda.max_memory_allocated(device),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


def _run_p0(args: argparse.Namespace) -> int:
    try:
        return run_p0(
            load_config(Path(args.paths)),
            load_experiment_config(Path(args.experiment)),
            load_gate_config(Path(args.gate)),
            Path(args.output),
            Path(args.manifest),
            Path(args.lock),
            {
                "paths": Path(args.paths),
                "experiment": Path(args.experiment),
                "gate": Path(args.gate),
            },
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True), flush=True)
        return 1


def _run_p1_smoke0(args: argparse.Namespace) -> int:
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    try:
        context = Smoke0Context(
            paths=load_config(paths_path),
            experiment=load_smoke0_config(experiment_path),
            manifest=Path(args.manifest),
            lock=Path(args.lock),
            output=Path(args.output),
            config_paths={"experiment": experiment_path, "paths": paths_path},
            cwd=Path.cwd(),
        )
        return run_smoke0(context)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True), flush=True)
        return 1


def _run_p2_a0_capacity(args: argparse.Namespace) -> int:
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    try:
        context = CapacityProbeContext(
            paths=load_config(paths_path),
            experiment=load_p2_a0_config(experiment_path),
            manifest=Path(args.manifest),
            lock=Path(args.lock),
            output=Path(args.output),
            config_paths={"experiment": experiment_path, "paths": paths_path},
            cwd=Path.cwd(),
        )
        return run_capacity_probe(context)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True),
            flush=True,
        )
        return 1


def _run_p2_a0_fewshot(args: argparse.Namespace) -> int:
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    try:
        context = FewshotContext(
            paths=load_config(paths_path),
            experiment=load_p2_a0_fewshot_config(experiment_path),
            manifest=Path(args.manifest),
            lock=Path(args.lock),
            output=Path(args.output),
            config_paths={"experiment": experiment_path, "paths": paths_path},
            cwd=Path.cwd(),
        )
        return run_fewshot(context)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True),
            flush=True,
        )
        return 1


def _run_p2_a0_single_type(args: argparse.Namespace) -> int:
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    try:
        context = FewshotContext(
            paths=load_config(paths_path),
            experiment=load_p2_a0_single_type_config(experiment_path),
            manifest=Path(args.manifest),
            lock=Path(args.lock),
            output=Path(args.output),
            config_paths={"experiment": experiment_path, "paths": paths_path},
            cwd=Path.cwd(),
        )
        return run_fewshot(context)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True),
            flush=True,
        )
        return 1


def _run_p2_a1_factorized(args: argparse.Namespace) -> int:
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    try:
        context = P2A1Context(
            paths=load_config(paths_path),
            experiment=load_p2_a1_config(experiment_path),
            manifest=Path(args.manifest),
            lock=Path(args.lock),
            output=Path(args.output),
            config_paths={"experiment": experiment_path, "paths": paths_path},
            cwd=Path.cwd(),
        )
        return run_p2_a1(context)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True),
            flush=True,
        )
        return 1


def _run_p2_b2_type_prompt_training(args: argparse.Namespace) -> int:
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    source_experiment_path = Path(args.source_experiment)
    try:
        context = P2B2Context(
            paths=load_config(paths_path),
            experiment=load_p2_b2_experiment_config(experiment_path),
            source_experiment=load_p2_a1_config(source_experiment_path),
            manifest=Path(args.manifest),
            lock=Path(args.lock),
            output=Path(args.output),
            config_paths={
                "experiment": experiment_path,
                "source_experiment": source_experiment_path,
                "paths": paths_path,
            },
            cwd=Path.cwd(),
        )
        return run_p2_b2(context)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True),
            flush=True,
        )
        return 1


def _run_v06_b_causal_renderer(args: argparse.Namespace) -> int:
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    source_model_path = Path(args.source_model)
    source_prompts_path = Path(args.source_prompts)
    try:
        experiment = load_v06_b_config(experiment_path)
        paths = load_config(paths_path)
        source_model = load_p2_a1_config(source_model_path)
        if isinstance(experiment, V06BMaskFirstConfig):
            source_prompts = load_mask_first_source_prompts(
                source_prompts_path,
                category=experiment.category,
            )
        else:
            source_prompts = load_p2_b2_config(source_prompts_path)
        if isinstance(experiment, V06BMaskFirstConfig):
            if args.attempt2_checkpoint is not None:
                raise ValueError("mask-first schema rejects --attempt2-checkpoint")
            parent_output = getattr(args, "parent_output", None)
            parent_experiment_path = getattr(args, "parent_experiment", None)
            context = MaskFirstV06BContext(
                paths=paths,
                experiment=experiment,
                source_model=source_model,
                source_prompts=source_prompts,
                manifest=Path(args.manifest),
                output=Path(args.output),
                cwd=Path.cwd(),
            )
            if args.mode == "train":
                if parent_output is not None or parent_experiment_path is not None:
                    raise ValueError(
                        "--parent-output is only valid for --mode continue-renderer, "
                        "--mode a-to-h, or --mode a2-to-h"
                    )
                run_mask_first_training(context)
            elif args.mode == "panel":
                if parent_output is not None or parent_experiment_path is not None:
                    raise ValueError(
                        "--parent-output is only valid for --mode continue-renderer, "
                        "--mode a-to-h, or --mode a2-to-h"
                    )
                run_mask_first_panel(context)
            elif args.mode == "continue-renderer":
                if parent_output is None or parent_experiment_path is None:
                    raise ValueError(
                        "continue-renderer requires --parent-output and --parent-experiment"
                    )
                parent_experiment = load_v06_b_config(Path(parent_experiment_path))
                if not isinstance(parent_experiment, V06BMaskFirstConfig):
                    raise ValueError("parent experiment must be a mask-first configuration")
                run_mask_first_renderer_continue(
                    context,
                    parent_output=Path(parent_output),
                    parent_experiment=parent_experiment,
                )
            elif args.mode == "a-to-h":
                if parent_output is None or parent_experiment_path is None:
                    raise ValueError("a-to-h requires --parent-output and --parent-experiment")
                parent_experiment = load_v06_b_config(Path(parent_experiment_path))
                if not isinstance(parent_experiment, V06BMaskFirstConfig):
                    raise ValueError("parent experiment must be a mask-first configuration")
                run_mask_first_a_to_h_training(
                    context,
                    parent_output=Path(parent_output),
                    parent_experiment=parent_experiment,
                )
            elif args.mode == "a2-to-h":
                if parent_output is None or parent_experiment_path is None:
                    raise ValueError("a2-to-h requires --parent-output and --parent-experiment")
                parent_experiment = load_v06_b_config(Path(parent_experiment_path))
                if not isinstance(parent_experiment, V06BMaskFirstConfig):
                    raise ValueError("parent experiment must be a mask-first configuration")
                run_mask_first_a2_to_h_training(
                    context,
                    parent_output=Path(parent_output),
                    parent_experiment=parent_experiment,
                )
            else:
                raise ValueError(
                    "mask-first schema allows only --mode train, --mode panel, "
                    "--mode continue-renderer, --mode a-to-h, or --mode a2-to-h"
                )
            return 0
        from cadiff.v06_b.runner import (
            V06BContext,
            run_v06_b_probe,
            run_v06_b_training,
        )

        context = V06BContext(
            paths=paths,
            experiment=experiment,
            source_model=source_model,
            source_prompts=source_prompts,
            manifest=Path(args.manifest),
            output=Path(args.output),
            cwd=Path.cwd(),
        )
        if args.mode != "matched-diagnostic" and args.attempt2_checkpoint is not None:
            raise ValueError("--attempt2-checkpoint is only valid for matched-diagnostic")
        if args.mode == "probe":
            run_v06_b_probe(context)
        elif args.mode == "train":
            run_v06_b_training(context)
        elif args.mode == "panel":
            from cadiff.v06_b.panel import run_v06_b_panel

            run_v06_b_panel(context)
        else:
            if args.attempt2_checkpoint is None:
                raise ValueError("matched-diagnostic requires --attempt2-checkpoint")
            from cadiff.v06_b.matched_diagnostic import run_matched_diagnostic

            run_matched_diagnostic(
                context,
                attempt2_checkpoint=Path(args.attempt2_checkpoint),
            )
        return 0
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True),
            flush=True,
        )
        return 1


def _run_p2_c0_hazelnut_downstream(args: argparse.Namespace) -> int:
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    source_experiment_path = Path(args.source_experiment)
    prompt_experiment_path = Path(args.prompt_experiment)
    renderer_manifest_path = Path(args.renderer_manifest)
    try:
        context = P2C0Context(
            paths=load_config(paths_path),
            experiment=load_p2_c0_config(experiment_path),
            source_experiment=load_p2_a1_config(source_experiment_path),
            prompt_experiment=load_p2_b2_config(prompt_experiment_path),
            renderer_manifest=renderer_manifest_path,
            manifest=Path(args.manifest),
            lock=Path(args.lock),
            output=Path(args.output),
            config_paths={
                "experiment": experiment_path,
                "source_experiment": source_experiment_path,
                "prompt_experiment": prompt_experiment_path,
                "renderer_manifest": renderer_manifest_path,
                "paths": paths_path,
            },
            cwd=Path.cwd(),
        )
        return run_p2_c0(context)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True),
            flush=True,
        )
        return 1


def _run_p3_a0_hazelnut_spatial_response(args: argparse.Namespace) -> int:
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    source_experiment_path = Path(args.source_experiment)
    generation_experiment_path = Path(args.generation_experiment)
    prompt_experiment_path = Path(args.prompt_experiment)
    try:
        context = P3A0Context(
            paths=load_config(paths_path),
            experiment=load_p3_a0_config(experiment_path),
            source_experiment=load_p2_a1_config(source_experiment_path),
            generation_experiment=load_p2_b1_config(generation_experiment_path),
            prompt_experiment=load_p2_b2_config(prompt_experiment_path),
            renderer_manifest=Path(args.renderer_manifest),
            normal_report=Path(args.normal_report),
            manifest=Path(args.manifest),
            lock=Path(args.lock),
            output=Path(args.output),
            config_paths={
                "paths": paths_path,
                "experiment": experiment_path,
                "source_experiment": source_experiment_path,
                "generation_experiment": generation_experiment_path,
                "prompt_experiment": prompt_experiment_path,
            },
            cwd=Path.cwd(),
        )
        return run_p3_a0(context)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True),
            flush=True,
        )
        return 1


def _run_p2_b3_checkpoint_paired_probe(args: argparse.Namespace) -> int:
    paths_path = Path(args.paths)
    generation_path = Path(args.generation)
    experiment_path = Path(args.experiment)
    source_experiment_path = Path(args.source_experiment)
    try:
        context = P2B3Context(
            paths=load_config(paths_path),
            generation=load_p2_b1_config(generation_path),
            experiment=load_p2_b3_config(experiment_path),
            source_experiment=load_p2_a1_config(source_experiment_path),
            manifest=Path(args.manifest),
            lock=Path(args.lock),
            source_run=Path(args.source_run),
            baseline_run=Path(args.baseline_run),
            reference_run=Path(args.reference_run),
            baseline_checkpoint=Path(args.baseline_checkpoint),
            candidate_checkpoint=Path(args.candidate_checkpoint),
            output=Path(args.output),
            config_paths={
                "paths": paths_path,
                "generation": generation_path,
                "experiment": experiment_path,
                "source_experiment": source_experiment_path,
            },
            cwd=Path.cwd(),
        )
        return run_p2_b3(context)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True),
            flush=True,
        )
        return 1


def _diagnose_gate_a(args: argparse.Namespace) -> int:
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    gate_path = Path(args.gate)
    diagnostic_path = Path(args.diagnostic)
    noise_replay_path = Path(args.noise_replay_config)
    try:
        config_paths = {
            "paths": paths_path,
            "experiment": experiment_path,
            "gate": gate_path,
            "diagnostic": diagnostic_path,
        }
        if args.phase in {
            "noise-replay-six-r1",
            "noise-replay-six-r2",
            "noise-replay-full-r1",
            "noise-replay-full-r2",
        }:
            load_noise_replay_config(noise_replay_path)
            config_paths["noise_replay"] = noise_replay_path
        context = DiagnosticContext(
            paths=load_config(paths_path),
            experiment=load_experiment_config(experiment_path),
            gate=load_gate_config(gate_path),
            diagnostic=load_diagnostic_config(diagnostic_path),
            output=Path(args.output),
            frozen_metrics=Path(args.frozen_metrics),
            manifest=Path(args.manifest),
            lock=Path(args.lock),
            config_paths=config_paths,
            cwd=Path.cwd(),
        )
        return run_diagnostic_phase(context, args.phase)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True), flush=True)
        return 1


def _run_diag_atlas_prop(args: argparse.Namespace) -> int:
    from cadiff.v06_atlas_diag.runner import load_atlas_diag_config, run_atlas_diag

    summary = run_atlas_diag(
        load_atlas_diag_config(Path(args.config)),
        repository_root=Path(args.repository_root),
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        output_relpath=args.output_relpath,
        device=args.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, sort_keys=True))
    return 0


def _run_atlas_a(args: argparse.Namespace) -> int:
    from cadiff.v06_a.a2 import main as atlas_a_main

    return atlas_a_main(
        [
            "--experiment",
            args.experiment,
            "--dataset-root",
            args.dataset_root,
            "--target-id",
            args.target_id,
            "--defect-type",
            args.defect_type,
            "--seed",
            str(args.seed),
            "--output",
            args.output,
            *(["--device", args.device] if args.device else []),
        ]
    )


def _run_diag_atlas_p1_recheck(args: argparse.Namespace) -> int:
    from cadiff.v06_atlas_diag.p1_recheck import load_p1_recheck_config, run_p1_recheck

    summary = run_p1_recheck(
        load_p1_recheck_config(Path(args.config), repository_root=Path(args.repository_root)),
        repository_root=Path(args.repository_root),
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        output_relpath=args.output_relpath,
        device=args.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, sort_keys=True))
    return 0


def _run_diag_atlas_a0v_compare(args: argparse.Namespace) -> int:
    from cadiff.v06_atlas_diag.a0v_compare import load_atlas_a0v_compare_config, run_atlas_a0v_compare

    summary = run_atlas_a0v_compare(
        load_atlas_a0v_compare_config(Path(args.config)),
        repository_root=Path(args.repository_root),
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        output_relpath=args.output_relpath,
        device=args.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, sort_keys=True))
    return 0


def _run_diag_atlas_cable_a(args: argparse.Namespace) -> int:
    from cadiff.v06_atlas_diag.cable_a import load_cable_a_config, run_cable_a_panel

    summary = run_cable_a_panel(
        load_cable_a_config(Path(args.config), repository_root=Path(args.repository_root)),
        repository_root=Path(args.repository_root),
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        output_relpath=args.output_relpath,
        device=args.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, sort_keys=True))
    return 0


def _run_diag_atlas_extra_a(args: argparse.Namespace) -> int:
    from cadiff.v06_atlas_diag.extra_a import run_extra_a

    summary = run_extra_a(
        mvtec_root=Path(args.mvtec_root),
        visa_root=Path(args.visa_root),
        output_root=Path(args.output_root),
        output_relpath=args.output_relpath,
        device=args.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, sort_keys=True))
    return 0


def _run_diag_atlas_cad3_a(args: argparse.Namespace) -> int:
    from cadiff.v06_atlas_diag.cad3_a import run_cad3_a

    summary = run_cad3_a(
        cad3_root=Path(args.cad3_root),
        output_root=Path(args.output_root),
        output_relpath=args.output_relpath,
        device=args.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, sort_keys=True))
    return 0


def _run_diag_a2_support(args: argparse.Namespace) -> int:
    from cadiff.v06_a.a2_support_diag import run_a2_support_diagnostic_from_cli

    report = run_a2_support_diagnostic_from_cli(
        paths=Path(args.paths),
        a_experiment=Path(args.a_experiment),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "identity_count": report["identity_count"],
                "failed_rows": report["failed_rows"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_h_corr3(args: argparse.Namespace) -> int:
    from cadiff.v06_b.h_corr3_diag import run_h_corr3_diagnostic_from_cli

    report = run_h_corr3_diagnostic_from_cli(
        paths=Path(args.paths),
        a_experiment=Path(args.a_experiment),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "identity_count": report["identity_count"],
                "corr_rows": report["corr_rows"],
                "rigid_fallback_rows": report["rigid_fallback_rows"],
                "failed_rows": report["failed_rows"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_h_corr3_extra(args: argparse.Namespace) -> int:
    from cadiff.v06_b.h_corr3_diag import run_h_corr3_extra_diagnostic_from_cli

    report = run_h_corr3_extra_diagnostic_from_cli(
        paths=Path(args.paths),
        visa_root=Path(args.visa_root),
        cad3_root=Path(args.cad3_root),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "identity_count": report["identity_count"],
                "corr_rows": report["corr_rows"],
                "rigid_fallback_rows": report["rigid_fallback_rows"],
                "failed_rows": report["failed_rows"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_h_pose_sample(args: argparse.Namespace) -> int:
    from cadiff.v06_b.h_pose_sample_diag import run_h_pose_sample_diagnostic_from_cli

    report = run_h_pose_sample_diagnostic_from_cli(
        paths=Path(args.paths),
        a_experiment=Path(args.a_experiment),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "identity_count": report["identity_count"],
                "diff_pose_rows": report["diff_pose_rows"],
                "same_pose_rows": report["same_pose_rows"],
                "floor_rows": report["floor_rows"],
                "failed_rows": report["failed_rows"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_h_pose_sample_extra(args: argparse.Namespace) -> int:
    from cadiff.v06_b.h_pose_sample_diag import run_h_pose_sample_extra_diagnostic_from_cli

    report = run_h_pose_sample_extra_diagnostic_from_cli(
        paths=Path(args.paths),
        visa_root=Path(args.visa_root),
        cad3_root=Path(args.cad3_root),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "identity_count": report["identity_count"],
                "diff_pose_rows": report["diff_pose_rows"],
                "same_pose_rows": report["same_pose_rows"],
                "floor_rows": report["floor_rows"],
                "failed_rows": report["failed_rows"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_a2_support_hard(args: argparse.Namespace) -> int:
    from cadiff.v06_a.a2_support_diag import run_a2_support_hard_diagnostic_from_cli

    report = run_a2_support_hard_diagnostic_from_cli(
        mvtec_root=Path(args.mvtec_root),
        visa_root=Path(args.visa_root),
        cad3_root=Path(args.cad3_root),
        can_root=Path(args.can_root),
        output=Path(args.output),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "identity_count": report["identity_count"],
                "failed_rows": report["failed_rows"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_wave1_label_geom(args: argparse.Namespace) -> int:
    from cadiff.v06_c.wave1_label_geom import run_wave1_label_geom_from_cli

    report = run_wave1_label_geom_from_cli(
        paths=Path(args.paths),
        a_experiment=Path(args.a_experiment),
        cable_dump=Path(args.cable_dump),
        can_dump=Path(args.can_dump),
        can_layout=Path(args.can_layout),
        visa_root=Path(args.visa_root),
        visa_dump_root=Path(args.visa_dump_root) if args.visa_dump_root else None,
        candle_dump=Path(args.candle_dump) if args.candle_dump else None,
        capsules_dump=Path(args.capsules_dump) if args.capsules_dump else None,
        cashew_dump=Path(args.cashew_dump) if args.cashew_dump else None,
        chewinggum_dump=Path(args.chewinggum_dump) if args.chewinggum_dump else None,
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "arm_names": report["arm_names"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_wave2_density_size(args: argparse.Namespace) -> int:
    from cadiff.v06_c.wave2_density_size import run_wave2_density_size_from_cli

    report = run_wave2_density_size_from_cli(
        paths=Path(args.paths),
        a_experiment=Path(args.a_experiment),
        cable_dump=Path(args.cable_dump),
        can_dump=Path(args.can_dump),
        can_layout=Path(args.can_layout),
        visa_root=Path(args.visa_root),
        visa_dump_root=Path(args.visa_dump_root) if args.visa_dump_root else None,
        candle_dump=Path(args.candle_dump) if args.candle_dump else None,
        capsules_dump=Path(args.capsules_dump) if args.capsules_dump else None,
        cashew_dump=Path(args.cashew_dump) if args.cashew_dump else None,
        chewinggum_dump=Path(args.chewinggum_dump) if args.chewinggum_dump else None,
        wave1_summary=Path(args.wave1_summary),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "arm_names": report["arm_names"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_v06_b_h_proposal_gap(args: argparse.Namespace) -> int:
    from cadiff.v06_b.h_proposal_gap import (
        run_h_proposal_gap_from_cli,
        run_h_proposal_gap_panels_from_cli,
    )

    common = dict(
        paths=Path(args.paths),
        experiment=Path(args.experiment),
        source_model=Path(args.source_model),
        source_prompts=Path(args.source_prompts),
        manifest=Path(args.manifest),
        generate_root=Path(args.generate_root),
        output=Path(args.output),
        cwd=Path.cwd(),
    )
    if getattr(args, "render_panel", False):
        manifest = run_h_proposal_gap_panels_from_cli(**common)
        print(json.dumps({"panel": manifest["files"], "output": args.output}, sort_keys=True))
        return 0
    report = run_h_proposal_gap_from_cli(**common)
    summary = {
        "schema_version": report["schema_version"],
        "train_unique_pairs": len(report["train"]["rows"]),
        "infer_identities": len(report["infer"]["rows"]),
        "output": args.output,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _run_v06_b_r_oracle_mask(args: argparse.Namespace) -> int:
    from cadiff.v06_b.r_oracle_mask import run_r_oracle_mask_from_cli

    report = run_r_oracle_mask_from_cli(
        paths=Path(args.paths),
        experiment=Path(args.experiment),
        source_model=Path(args.source_model),
        source_prompts=Path(args.source_prompts),
        manifest=Path(args.manifest),
        generate_config=Path(args.generate_config),
        generate_root=Path(args.generate_root),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    summary = {
        "schema_version": report["schema_version"],
        "diagnostic_only": True,
        "identity_count": report["summary"]["identity_count"],
        "oracle_empty": report["summary"]["oracle_empty"],
        "output": args.output,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _run_v06_b_a2_shape_h(args: argparse.Namespace) -> int:
    from cadiff.v06_b.a2_shape_instantiate import run_a2_shape_instantiate_from_cli

    report = run_a2_shape_instantiate_from_cli(
        paths=Path(args.paths),
        a_experiment=Path(args.a_experiment),
        generate_config=Path(args.generate_config),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        pixel_h_root=Path(args.pixel_h_root) if args.pixel_h_root else None,
        device=args.device,
    )
    summary = {
        "schema_version": report["schema_version"],
        "diagnostic_only": True,
        "identity_count": report["identity_count"],
        "failed_rows": report["failed_rows"],
        "output": args.output,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _run_v06_b_a2_cc_place(args: argparse.Namespace) -> int:
    from cadiff.v06_b.a2_component_place import run_a2_component_place_from_cli

    report = run_a2_component_place_from_cli(
        paths=Path(args.paths),
        a_experiment=Path(args.a_experiment),
        generate_config=Path(args.generate_config),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        pixel_h_root=Path(args.pixel_h_root) if args.pixel_h_root else None,
        device=args.device,
    )
    summary = {
        "schema_version": report["schema_version"],
        "diagnostic_only": True,
        "identity_count": report["identity_count"],
        "failed_rows": report["failed_rows"],
        "floor_rows": report.get("floor_rows", 0),
        "output": args.output,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _run_v06_b_a2_cc_place_extra(args: argparse.Namespace) -> int:
    from cadiff.v06_b.a2_component_place_extra import run_a2_component_place_extra_from_cli

    report = run_a2_component_place_extra_from_cli(
        paths=Path(args.paths),
        visa_root=Path(args.visa_root),
        cad3_root=Path(args.cad3_root),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    summary = {
        "schema_version": report["schema_version"],
        "diagnostic_only": True,
        "identity_count": report["identity_count"],
        "failed_rows": report["failed_rows"],
        "floor_rows": report.get("floor_rows", 0),
        "output": args.output,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _run_v06_c_cable_place_bulk(args: argparse.Namespace) -> int:
    from cadiff.v06_c.cable_place_bulk import run_cable_place_bulk_from_cli

    report = run_cable_place_bulk_from_cli(
        paths=Path(args.paths),
        a_experiment=Path(args.a_experiment),
        generate_config=Path(args.generate_config),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
        pairs_per_type=int(args.pairs_per_type),
        index_start=int(args.index_start),
    )
    summary = {
        "schema_version": report["schema_version"],
        "diagnostic_only": True,
        "identity_count": report["identity_count"],
        "kept_rows": report["kept_rows"],
        "failed_rows": report["failed_rows"],
        "output": args.output,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _run_v06_c_can_train_r(args: argparse.Namespace) -> int:
    from cadiff.v06_c.can_exact_mask_r import train_can_exact_mask_r_from_cli

    report = train_can_exact_mask_r_from_cli(
        paths=Path(args.paths),
        manifest=Path(args.manifest),
        dataset_root=Path(args.dataset_root),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    print(json.dumps({key: report[key] for key in (
        "schema_version",
        "diagnostic_only",
        "steps",
        "fewshot_train",
        "last_total",
        "output",
    )}, sort_keys=True))
    return 0


def _run_v06_c_can_place_bulk(args: argparse.Namespace) -> int:
    from cadiff.v06_c.can_place_bulk import run_can_place_bulk_from_cli

    report = run_can_place_bulk_from_cli(
        paths=Path(args.paths),
        source_root=Path(args.source_root),
        layout_root=Path(args.layout_root),
        checkpoint=Path(args.checkpoint),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
        pairs=int(args.pairs),
    )
    summary = {
        "schema_version": report["schema_version"],
        "diagnostic_only": True,
        "identity_count": report["identity_count"],
        "kept_rows": report["kept_rows"],
        "failed_rows": report["failed_rows"],
        "output": args.output,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _run_v06_c_can_train_native_win_r(args: argparse.Namespace) -> int:
    from cadiff.v06_c.can_native_win_r import train_can_native_win_r_from_cli

    report = train_can_native_win_r_from_cli(
        paths=Path(args.paths),
        manifest=Path(args.manifest),
        dataset_root=Path(args.dataset_root),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    print(json.dumps({key: report[key] for key in (
        "schema_version",
        "diagnostic_only",
        "steps",
        "fewshot_train",
        "last_total",
        "output",
    )}, sort_keys=True))
    return 0


def _run_v06_c_can_native_win_bulk(args: argparse.Namespace) -> int:
    from cadiff.v06_c.can_native_win_bulk import run_can_native_win_bulk_from_cli

    report = run_can_native_win_bulk_from_cli(
        paths=Path(args.paths),
        source_root=Path(args.source_root),
        layout_root=Path(args.layout_root),
        checkpoint=Path(args.checkpoint),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    summary = {
        "schema_version": report["schema_version"],
        "diagnostic_only": True,
        "identity_count": report["identity_count"],
        "kept_rows": report["kept_rows"],
        "failed_rows": report["failed_rows"],
        "output": args.output,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _run_v06_c_visa_train_r(args: argparse.Namespace) -> int:
    from cadiff.v06_c.visa_exact_mask_r import train_visa_exact_mask_r_from_cli

    report = train_visa_exact_mask_r_from_cli(
        paths=Path(args.paths),
        manifest=Path(args.manifest),
        dataset_root=Path(args.dataset_root),
        output=Path(args.output),
        category=args.category,
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
        collapse_type=args.collapse_type,
        split=args.split,
    )
    print(json.dumps({key: report[key] for key in (
        "schema_version",
        "diagnostic_only",
        "category",
        "steps",
        "fewshot_train",
        "last_total",
        "output",
    )}, sort_keys=True))
    return 0


def _run_v06_c_visa_place_bulk(args: argparse.Namespace) -> int:
    from cadiff.v06_c.visa_place_bulk import run_visa_place_bulk_from_cli

    report = run_visa_place_bulk_from_cli(
        paths=Path(args.paths),
        source_root=Path(args.source_root),
        category=args.category,
        checkpoint=Path(args.checkpoint),
        output=Path(args.output),
        cwd=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
        pairs_per_type=int(args.pairs_per_type),
        index_start=int(args.index_start),
        collapse_type=args.collapse_type,
        split=args.split,
        only_types=tuple(args.only_types) if args.only_types else None,
        combined_large_cluster_area=bool(args.combined_large_cluster_area),
        combined_large_cluster_source=bool(args.combined_large_cluster_source),
        combined_mix_y_stamps=bool(args.combined_mix_y_stamps),
    )
    summary = {
        "schema_version": report["schema_version"],
        "diagnostic_only": True,
        "category": report["category"],
        "identity_count": report["identity_count"],
        "kept_rows": report["kept_rows"],
        "failed_rows": report["failed_rows"],
        "floor_rows": report["floor_rows"],
        "output": args.output,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _run_diag_tokencut_normals(args: argparse.Namespace) -> int:
    from cadiff.v06_c.tokencut_normal_preview import run_tokencut_normal_preview_from_cli

    report = run_tokencut_normal_preview_from_cli(
        mvtec_root=Path(args.mvtec_root),
        visa_root=Path(args.visa_root),
        cable_experiment=Path(args.cable_experiment),
        output=Path(args.output),
        repository_root=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "arm_names": report["arm_names"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_tokencut_relarea(args: argparse.Namespace) -> int:
    from cadiff.v06_c.tokencut_relarea_preview import run_tokencut_relarea_preview_from_cli

    report = run_tokencut_relarea_preview_from_cli(
        identities_path=Path(args.identities),
        mvtec_root=Path(args.mvtec_root),
        visa_root=Path(args.visa_root),
        cable_experiment=Path(args.cable_experiment),
        output=Path(args.output),
        repository_root=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
        tau=float(args.tau),
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "tau": report["tau"],
                "arm_names": report["arm_names"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_a2_quantile_cover(args: argparse.Namespace) -> int:
    from cadiff.v06_a.a2_quantile_cover_diag import run_a2_quantile_cover_preview_from_cli

    report = run_a2_quantile_cover_preview_from_cli(
        mvtec_root=Path(args.mvtec_root),
        visa_root=Path(args.visa_root),
        cable_experiment=Path(args.cable_experiment),
        output=Path(args.output),
        repository_root=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "n": report["n"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_h_site_scale_coverage(args: argparse.Namespace) -> int:
    from cadiff.v06_b.h_site_scale_diag import run_h_site_scale_coverage_from_cli

    report = run_h_site_scale_coverage_from_cli(
        mvtec_root=Path(args.mvtec_root),
        visa_root=Path(args.visa_root),
        cable_experiment=Path(args.cable_experiment),
        output=Path(args.output),
        repository_root=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
        n_draws=int(args.n_draws),
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "n": report["n"],
                "n_draws": report["n_draws"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_proposal_tokencut_cc(args: argparse.Namespace) -> int:
    from cadiff.v06_c.proposal_on_tokencut_cc import run_proposal_on_tokencut_cc_preview_from_cli

    report = run_proposal_on_tokencut_cc_preview_from_cli(
        mvtec_root=Path(args.mvtec_root),
        visa_root=Path(args.visa_root),
        cable_experiment=Path(args.cable_experiment),
        output=Path(args.output),
        repository_root=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "n": report["n"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_h_scale_clip(args: argparse.Namespace) -> int:
    from cadiff.v06_b.h_scale_clip_diag import run_h_scale_clip_preview_from_cli

    report = run_h_scale_clip_preview_from_cli(
        mvtec_root=Path(args.mvtec_root),
        visa_root=Path(args.visa_root),
        cable_experiment=Path(args.cable_experiment),
        output=Path(args.output),
        repository_root=Path(args.repository_root) if args.repository_root else Path.cwd(),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "n": report["n"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_loc_pred_gt_overlay(args: argparse.Namespace) -> int:
    from cadiff.v06_c.loc_pred_gt_overlay import run_loc_pred_gt_overlay_from_cli

    report = run_loc_pred_gt_overlay_from_cli(
        eval_root=Path(args.eval_root),
        mvtec_path=Path(args.mvtec_path),
        eval_kit=Path(args.eval_kit),
        output=Path(args.output),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "arm_names": report["arm_names"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_loc_score_decomp(args: argparse.Namespace) -> int:
    from cadiff.v06_c.loc_score_decomp import run_loc_score_decomp_from_cli

    report = run_loc_score_decomp_from_cli(
        eval_root=Path(args.eval_root),
        mvtec_path=Path(args.mvtec_path),
        eval_kit=Path(args.eval_kit),
        output=Path(args.output),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "no_can": True,
                "arm_names": report["arm_names"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_seas_pixel_ap_buckets(args: argparse.Namespace) -> int:
    from cadiff.v06_c.seas_pixel_ap_buckets import run_seas_pixel_ap_buckets_from_cli

    report = run_seas_pixel_ap_buckets_from_cli(
        eval_root=Path(args.eval_root),
        eval_kit=Path(args.eval_kit),
        output=Path(args.output),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "no_can": True,
                "not_train_r": True,
                "arm_names": report["arm_names"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_oneshot_appearance(args: argparse.Namespace) -> int:
    from cadiff.v06_c.oneshot_appearance_panel import run_oneshot_appearance_panel_from_cli

    report = run_oneshot_appearance_panel_from_cli(
        runs_v06c=Path(args.runs_root),
        output=Path(args.output),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "no_can": True,
                "n": report["n"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_visa_type_gt_mfinal(args: argparse.Namespace) -> int:
    from cadiff.v06_c.visa_type_gt_mfinal_panel import run_visa_type_gt_mfinal_panel_from_cli

    report = run_visa_type_gt_mfinal_panel_from_cli(
        eval_root=Path(args.eval_root),
        output=Path(args.output),
        category=str(args.category),
        type_slugs=tuple(args.type_slug),
        n_rows_per_type=int(args.n_rows_per_type),
        eval_name=str(args.eval_name),
        layout_name=str(args.layout_name),
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "category": report["category"],
                "diagnostic_only": True,
                "n": report["n"],
                "no_can": True,
                "not_train_r": True,
                "output": args.output,
                "type_slugs": report["type_slugs"],
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_capsules_combined_panel(args: argparse.Namespace) -> int:
    from cadiff.v06_c.capsules_combined_panel import run_capsules_combined_panel_from_cli

    report = run_capsules_combined_panel_from_cli(
        eval_root=Path(args.eval_root),
        output=Path(args.output),
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "no_can": True,
                "not_train_r": True,
                "n": report["n"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_capsules_combined_upper_area(args: argparse.Namespace) -> int:
    from cadiff.v06_c.capsules_combined_upper_area_panel import (
        run_capsules_combined_upper_area_from_cli,
    )
    from cadiff.v06_c.visa_pilot import SOURCE_ROOT

    report = run_capsules_combined_upper_area_from_cli(
        eval_root=Path(args.eval_root),
        output=Path(args.output),
        source_root=Path(args.source_root) if args.source_root else SOURCE_ROOT,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "not_official_h": True,
                "no_can": True,
                "not_train_r": True,
                "n": report["n"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_diag_capsules_combined_two_stamp(args: argparse.Namespace) -> int:
    from cadiff.v06_c.capsules_combined_two_stamp_panel import (
        run_capsules_combined_two_stamp_from_cli,
    )
    from cadiff.v06_c.visa_pilot import SOURCE_ROOT

    report = run_capsules_combined_two_stamp_from_cli(
        eval_root=Path(args.eval_root),
        output=Path(args.output),
        source_root=Path(args.source_root) if args.source_root else SOURCE_ROOT,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "not_official_h": True,
                "no_can": True,
                "not_train_r": True,
                "n": report["n"],
                "n_two_stamps": report["n_two_stamps"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_official_generate(args: argparse.Namespace) -> int:
    import yaml
    schema = yaml.safe_load(Path(args.config).read_text()).get("schema_version")
    if schema == "v06_c_cable_generate":
        from cadiff.v06_c.cable_generate import load_cable_generate_config, run_cable_generate_panel
        summary = run_cable_generate_panel(
            load_cable_generate_config(Path(args.config), repository_root=Path(args.repository_root)),
            repository_root=Path(args.repository_root), dataset_root=Path(args.dataset_root),
            output_root=Path(args.output_root), output_relpath=args.output_relpath, device=args.device,
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    from cadiff.v06_c.generate import load_generate_config, run_official_generate_panel

    summary = run_official_generate_panel(
        load_generate_config(Path(args.config), repository_root=Path(args.repository_root)),
        repository_root=Path(args.repository_root),
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        output_relpath=args.output_relpath,
        device=args.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, sort_keys=True))
    return 0


def _run_diag_atlas_p2_cascade(args: argparse.Namespace) -> int:
    from cadiff.v06_atlas_diag.p2_cascade import load_p2_cascade_config, run_p2_cascade

    summary = run_p2_cascade(
        load_p2_cascade_config(Path(args.config), repository_root=Path(args.repository_root)),
        repository_root=Path(args.repository_root),
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        output_relpath=args.output_relpath,
        device=args.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cadiff")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--split-id", required=True)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(handler=_build_manifest)
    check_model = subparsers.add_parser("check-model")
    check_model.add_argument("--config", required=True)
    check_model.set_defaults(handler=_check_model)
    p0 = subparsers.add_parser("run-p0")
    p0.add_argument("--paths", required=True)
    p0.add_argument("--experiment", required=True)
    p0.add_argument("--gate", required=True)
    p0.add_argument("--output", required=True)
    p0.add_argument(
        "--manifest",
        default="manifests/mvtec_lexicographic_floor_third_v1.jsonl",
    )
    p0.add_argument("--lock", default="requirements/lock.omen-home.txt")
    p0.add_argument("--overwrite", action="store_true")
    p0.set_defaults(handler=_run_p0)
    p1_smoke0 = subparsers.add_parser("p1-smoke0")
    p1_smoke0.add_argument("--paths", required=True)
    p1_smoke0.add_argument("--experiment", required=True)
    p1_smoke0.add_argument("--manifest", required=True)
    p1_smoke0.add_argument("--lock", required=True)
    p1_smoke0.add_argument("--output", required=True)
    p1_smoke0.set_defaults(handler=_run_p1_smoke0)
    p2_capacity = subparsers.add_parser("p2-a0-capacity")
    p2_capacity.add_argument("--paths", required=True)
    p2_capacity.add_argument("--experiment", required=True)
    p2_capacity.add_argument("--manifest", required=True)
    p2_capacity.add_argument("--lock", required=True)
    p2_capacity.add_argument("--output", required=True)
    p2_capacity.set_defaults(handler=_run_p2_a0_capacity)
    p2_fewshot = subparsers.add_parser("p2-a0-fewshot")
    p2_fewshot.add_argument("--paths", required=True)
    p2_fewshot.add_argument("--experiment", required=True)
    p2_fewshot.add_argument("--manifest", required=True)
    p2_fewshot.add_argument("--lock", required=True)
    p2_fewshot.add_argument("--output", required=True)
    p2_fewshot.set_defaults(handler=_run_p2_a0_fewshot)
    p2_single_type = subparsers.add_parser("p2-a0-single-type")
    p2_single_type.add_argument("--paths", required=True)
    p2_single_type.add_argument("--experiment", required=True)
    p2_single_type.add_argument("--manifest", required=True)
    p2_single_type.add_argument("--lock", required=True)
    p2_single_type.add_argument("--output", required=True)
    p2_single_type.set_defaults(handler=_run_p2_a0_single_type)
    p2_factorized = subparsers.add_parser("p2-a1-factorized")
    p2_factorized.add_argument("--paths", required=True)
    p2_factorized.add_argument("--experiment", required=True)
    p2_factorized.add_argument("--manifest", required=True)
    p2_factorized.add_argument("--lock", required=True)
    p2_factorized.add_argument("--output", required=True)
    p2_factorized.set_defaults(handler=_run_p2_a1_factorized)
    p2_type_prompt_training = subparsers.add_parser("p2-b2-type-prompt-training")
    p2_type_prompt_training.add_argument("--paths", required=True)
    p2_type_prompt_training.add_argument("--experiment", required=True)
    p2_type_prompt_training.add_argument("--source-experiment", required=True)
    p2_type_prompt_training.add_argument("--manifest", required=True)
    p2_type_prompt_training.add_argument("--lock", required=True)
    p2_type_prompt_training.add_argument("--output", required=True)
    p2_type_prompt_training.set_defaults(handler=_run_p2_b2_type_prompt_training)
    v06_b = subparsers.add_parser("v06-b-causal-renderer")
    v06_b.add_argument("--paths", required=True)
    v06_b.add_argument("--experiment", required=True)
    v06_b.add_argument("--source-model", required=True)
    v06_b.add_argument("--source-prompts", required=True)
    v06_b.add_argument("--manifest", required=True)
    v06_b.add_argument("--output", required=True)
    v06_b.add_argument(
        "--mode",
        required=True,
        choices=(
            "probe",
            "train",
            "panel",
            "matched-diagnostic",
            "continue-renderer",
            "a-to-h",
            "a2-to-h",
        ),
    )
    v06_b.add_argument("--attempt2-checkpoint")
    v06_b.add_argument("--parent-output")
    v06_b.add_argument("--parent-experiment")
    v06_b.set_defaults(handler=_run_v06_b_causal_renderer)
    v06_b_gap = subparsers.add_parser("v06-b-h-proposal-gap")
    v06_b_gap.add_argument("--paths", required=True)
    v06_b_gap.add_argument("--experiment", required=True)
    v06_b_gap.add_argument("--source-model", required=True)
    v06_b_gap.add_argument("--source-prompts", required=True)
    v06_b_gap.add_argument("--manifest", required=True)
    v06_b_gap.add_argument("--generate-root", required=True)
    v06_b_gap.add_argument("--output", required=True)
    v06_b_gap.add_argument("--render-panel", action="store_true")
    v06_b_gap.set_defaults(handler=_run_v06_b_h_proposal_gap)
    v06_b_oracle = subparsers.add_parser("v06-b-r-oracle-mask")
    v06_b_oracle.add_argument("--paths", required=True)
    v06_b_oracle.add_argument("--experiment", required=True)
    v06_b_oracle.add_argument("--source-model", required=True)
    v06_b_oracle.add_argument("--source-prompts", required=True)
    v06_b_oracle.add_argument("--manifest", required=True)
    v06_b_oracle.add_argument("--generate-config", required=True)
    v06_b_oracle.add_argument("--generate-root", required=True)
    v06_b_oracle.add_argument("--output", required=True)
    v06_b_oracle.add_argument("--repository-root")
    v06_b_oracle.add_argument("--device")
    v06_b_oracle.set_defaults(handler=_run_v06_b_r_oracle_mask)
    v06_b_shape = subparsers.add_parser("v06-b-a2-shape-h")
    v06_b_shape.add_argument("--paths", required=True)
    v06_b_shape.add_argument("--a-experiment", required=True)
    v06_b_shape.add_argument("--generate-config", required=True)
    v06_b_shape.add_argument("--output", required=True)
    v06_b_shape.add_argument("--pixel-h-root")
    v06_b_shape.add_argument("--repository-root")
    v06_b_shape.add_argument("--device")
    v06_b_shape.set_defaults(handler=_run_v06_b_a2_shape_h)
    v06_b_cc = subparsers.add_parser("v06-b-a2-cc-place")
    v06_b_cc.add_argument("--paths", required=True)
    v06_b_cc.add_argument("--a-experiment", required=True)
    v06_b_cc.add_argument("--generate-config", required=True)
    v06_b_cc.add_argument("--output", required=True)
    v06_b_cc.add_argument("--pixel-h-root")
    v06_b_cc.add_argument("--repository-root")
    v06_b_cc.add_argument("--device")
    v06_b_cc.set_defaults(handler=_run_v06_b_a2_cc_place)
    v06_b_cc_extra = subparsers.add_parser("v06-b-a2-cc-place-extra")
    v06_b_cc_extra.add_argument("--paths", required=True)
    v06_b_cc_extra.add_argument("--visa-root", required=True)
    v06_b_cc_extra.add_argument("--cad3-root", required=True)
    v06_b_cc_extra.add_argument("--output", required=True)
    v06_b_cc_extra.add_argument("--repository-root")
    v06_b_cc_extra.add_argument("--device")
    v06_b_cc_extra.set_defaults(handler=_run_v06_b_a2_cc_place_extra)
    v06_c_bulk = subparsers.add_parser("v06-c-cable-place-bulk")
    v06_c_bulk.add_argument("--paths", required=True)
    v06_c_bulk.add_argument("--a-experiment", required=True)
    v06_c_bulk.add_argument("--generate-config", required=True)
    v06_c_bulk.add_argument("--output", required=True)
    v06_c_bulk.add_argument("--repository-root")
    v06_c_bulk.add_argument("--device")
    v06_c_bulk.add_argument("--pairs-per-type", type=int, default=500)
    v06_c_bulk.add_argument("--index-start", type=int, default=0)
    v06_c_bulk.set_defaults(handler=_run_v06_c_cable_place_bulk)
    v06_c_can_r = subparsers.add_parser("v06-c-can-train-r")
    v06_c_can_r.add_argument("--paths", required=True)
    v06_c_can_r.add_argument("--manifest", required=True)
    v06_c_can_r.add_argument("--dataset-root", required=True)
    v06_c_can_r.add_argument("--output", required=True)
    v06_c_can_r.add_argument("--repository-root")
    v06_c_can_r.add_argument("--device")
    v06_c_can_r.set_defaults(handler=_run_v06_c_can_train_r)
    v06_c_can_bulk = subparsers.add_parser("v06-c-can-place-bulk")
    v06_c_can_bulk.add_argument("--paths", required=True)
    v06_c_can_bulk.add_argument("--source-root", required=True)
    v06_c_can_bulk.add_argument("--layout-root", required=True)
    v06_c_can_bulk.add_argument("--checkpoint", required=True)
    v06_c_can_bulk.add_argument("--output", required=True)
    v06_c_can_bulk.add_argument("--repository-root")
    v06_c_can_bulk.add_argument("--device")
    v06_c_can_bulk.add_argument("--pairs", type=int, default=500)
    v06_c_can_bulk.set_defaults(handler=_run_v06_c_can_place_bulk)
    v06_c_can_native_r = subparsers.add_parser("v06-c-can-train-native-win-r")
    v06_c_can_native_r.add_argument("--paths", required=True)
    v06_c_can_native_r.add_argument("--manifest", required=True)
    v06_c_can_native_r.add_argument("--dataset-root", required=True)
    v06_c_can_native_r.add_argument("--output", required=True)
    v06_c_can_native_r.add_argument("--repository-root")
    v06_c_can_native_r.add_argument("--device")
    v06_c_can_native_r.set_defaults(handler=_run_v06_c_can_train_native_win_r)
    v06_c_can_native_bulk = subparsers.add_parser("v06-c-can-native-win-bulk")
    v06_c_can_native_bulk.add_argument("--paths", required=True)
    v06_c_can_native_bulk.add_argument("--source-root", required=True)
    v06_c_can_native_bulk.add_argument("--layout-root", required=True)
    v06_c_can_native_bulk.add_argument("--checkpoint", required=True)
    v06_c_can_native_bulk.add_argument("--output", required=True)
    v06_c_can_native_bulk.add_argument("--repository-root")
    v06_c_can_native_bulk.add_argument("--device")
    v06_c_can_native_bulk.set_defaults(handler=_run_v06_c_can_native_win_bulk)
    v06_c_visa_r = subparsers.add_parser("v06-c-visa-train-r")
    v06_c_visa_r.add_argument("--paths", required=True)
    v06_c_visa_r.add_argument("--manifest", required=True)
    v06_c_visa_r.add_argument("--dataset-root", required=True)
    v06_c_visa_r.add_argument("--output", required=True)
    v06_c_visa_r.add_argument("--category", required=True)
    v06_c_visa_r.add_argument("--repository-root")
    v06_c_visa_r.add_argument("--device")
    v06_c_visa_r.add_argument("--collapse-type", default=None, choices=("defect",))
    v06_c_visa_r.add_argument(
        "--split", default="extra-a", choices=("extra-a", "defect", "seas", "magic", "mvtec")
    )
    v06_c_visa_r.set_defaults(handler=_run_v06_c_visa_train_r)
    v06_c_visa_bulk = subparsers.add_parser("v06-c-visa-place-bulk")
    v06_c_visa_bulk.add_argument("--paths", required=True)
    v06_c_visa_bulk.add_argument("--source-root", required=True)
    v06_c_visa_bulk.add_argument("--category", required=True)
    v06_c_visa_bulk.add_argument("--checkpoint", required=True)
    v06_c_visa_bulk.add_argument("--output", required=True)
    v06_c_visa_bulk.add_argument("--repository-root")
    v06_c_visa_bulk.add_argument("--device")
    v06_c_visa_bulk.add_argument("--pairs-per-type", type=int, default=500)
    v06_c_visa_bulk.add_argument("--index-start", type=int, default=0)
    v06_c_visa_bulk.add_argument("--collapse-type", default=None, choices=("defect",))
    v06_c_visa_bulk.add_argument(
        "--split", default="extra-a", choices=("extra-a", "defect", "seas", "magic", "mvtec")
    )
    v06_c_visa_bulk.add_argument("--only-types", nargs="+")
    v06_c_visa_bulk.add_argument(
        "--combined-large-cluster-area",
        action="store_true",
        help="diagnostic: combined A* from the high few-shot mode after the largest log-gap",
    )
    v06_c_visa_bulk.add_argument(
        "--combined-large-cluster-source",
        action="store_true",
        help="diagnostic: combined few-shot sources restricted to that high mode",
    )
    v06_c_visa_bulk.add_argument(
        "--combined-mix-y-stamps",
        action="store_true",
        help="diagnostic: combined y=1 or 2 disjoint stamps from the job seed",
    )
    v06_c_visa_bulk.set_defaults(handler=_run_v06_c_visa_place_bulk)
    p2_downstream = subparsers.add_parser("p2-c0-hazelnut-downstream")
    p2_downstream.add_argument("--paths", required=True)
    p2_downstream.add_argument("--experiment", required=True)
    p2_downstream.add_argument("--source-experiment", required=True)
    p2_downstream.add_argument("--prompt-experiment", required=True)
    p2_downstream.add_argument("--renderer-manifest", required=True)
    p2_downstream.add_argument("--manifest", required=True)
    p2_downstream.add_argument("--lock", required=True)
    p2_downstream.add_argument("--output", required=True)
    p2_downstream.set_defaults(handler=_run_p2_c0_hazelnut_downstream)
    p3_spatial_response = subparsers.add_parser("p3-a0-hazelnut-spatial-response")
    p3_spatial_response.add_argument("--paths", required=True)
    p3_spatial_response.add_argument("--experiment", required=True)
    p3_spatial_response.add_argument("--source-experiment", required=True)
    p3_spatial_response.add_argument("--generation-experiment", required=True)
    p3_spatial_response.add_argument("--prompt-experiment", required=True)
    p3_spatial_response.add_argument("--renderer-manifest", required=True)
    p3_spatial_response.add_argument("--normal-report", required=True)
    p3_spatial_response.add_argument("--manifest", required=True)
    p3_spatial_response.add_argument("--lock", required=True)
    p3_spatial_response.add_argument("--output", required=True)
    p3_spatial_response.set_defaults(handler=_run_p3_a0_hazelnut_spatial_response)
    p2_checkpoint_probe = subparsers.add_parser("p2-b3-checkpoint-paired-probe")
    p2_checkpoint_probe.add_argument("--paths", required=True)
    p2_checkpoint_probe.add_argument("--generation", required=True)
    p2_checkpoint_probe.add_argument("--experiment", required=True)
    p2_checkpoint_probe.add_argument("--source-experiment", required=True)
    p2_checkpoint_probe.add_argument("--manifest", required=True)
    p2_checkpoint_probe.add_argument("--lock", required=True)
    p2_checkpoint_probe.add_argument("--source-run", required=True)
    p2_checkpoint_probe.add_argument("--baseline-run", required=True)
    p2_checkpoint_probe.add_argument("--reference-run", required=True)
    p2_checkpoint_probe.add_argument("--baseline-checkpoint", required=True)
    p2_checkpoint_probe.add_argument("--candidate-checkpoint", required=True)
    p2_checkpoint_probe.add_argument("--output", required=True)
    p2_checkpoint_probe.set_defaults(handler=_run_p2_b3_checkpoint_paired_probe)
    diagnostic = subparsers.add_parser("diagnose-gate-a")
    diagnostic.add_argument("--paths", required=True)
    diagnostic.add_argument("--experiment", required=True)
    diagnostic.add_argument("--gate", required=True)
    diagnostic.add_argument("--diagnostic", required=True)
    diagnostic.add_argument(
        "--phase",
        required=True,
        choices=(
            "baseline-repeat",
            "single-factor",
            "minimal-combinations",
            "full-candidate",
            "noise-replay-six-r1",
            "noise-replay-six-r2",
            "noise-replay-full-r1",
            "noise-replay-full-r2",
        ),
    )
    diagnostic.add_argument(
        "--noise-replay-config",
        default="configs/diagnostics/gate_a_noise_replay_v1.yaml",
    )
    diagnostic.add_argument("--output", required=True)
    diagnostic.add_argument("--frozen-metrics", required=True)
    diagnostic.add_argument(
        "--manifest",
        default="manifests/mvtec_lexicographic_floor_third_v1.jsonl",
    )
    diagnostic.add_argument("--lock", default="requirements/lock.omen-home.txt")
    diagnostic.set_defaults(handler=_diagnose_gate_a)
    atlas = subparsers.add_parser("diag-atlas-prop")
    atlas.add_argument("--config", required=True)
    atlas.add_argument("--repository-root", required=True)
    atlas.add_argument("--dataset-root", required=True)
    atlas.add_argument("--output-root", required=True)
    atlas.add_argument("--output-relpath", required=True)
    atlas.add_argument("--device")
    atlas.set_defaults(handler=_run_diag_atlas_prop)
    atlas_a0v = subparsers.add_parser("diag-atlas-a0v-compare")
    atlas_a0v.add_argument("--config", required=True)
    atlas_a0v.add_argument("--repository-root", required=True)
    atlas_a0v.add_argument("--dataset-root", required=True)
    atlas_a0v.add_argument("--output-root", required=True)
    atlas_a0v.add_argument("--output-relpath", required=True)
    atlas_a0v.add_argument("--device")
    atlas_a0v.set_defaults(handler=_run_diag_atlas_a0v_compare)
    atlas_p1 = subparsers.add_parser("diag-atlas-p1-recheck")
    atlas_p1.add_argument("--config", required=True)
    atlas_p1.add_argument("--repository-root", required=True)
    atlas_p1.add_argument("--dataset-root", required=True)
    atlas_p1.add_argument("--output-root", required=True)
    atlas_p1.add_argument("--output-relpath", required=True)
    atlas_p1.add_argument("--device")
    atlas_p1.set_defaults(handler=_run_diag_atlas_p1_recheck)
    atlas_a = subparsers.add_parser("atlas-a")
    atlas_a.add_argument("--experiment", required=True)
    atlas_a.add_argument("--dataset-root", required=True)
    atlas_a.add_argument("--target-id", required=True)
    atlas_a.add_argument("--defect-type", required=True)
    atlas_a.add_argument("--seed", type=int, required=True)
    atlas_a.add_argument("--output", required=True)
    atlas_a.add_argument("--device")
    atlas_a.set_defaults(handler=_run_atlas_a)
    atlas_p2 = subparsers.add_parser("diag-atlas-p2-cascade")
    atlas_p2.add_argument("--config", required=True)
    atlas_p2.add_argument("--repository-root", required=True)
    atlas_p2.add_argument("--dataset-root", required=True)
    atlas_p2.add_argument("--output-root", required=True)
    atlas_p2.add_argument("--output-relpath", required=True)
    atlas_p2.add_argument("--device")
    atlas_p2.set_defaults(handler=_run_diag_atlas_p2_cascade)
    atlas_cable_a = subparsers.add_parser("diag-atlas-cable-a")
    atlas_cable_a.add_argument("--config", required=True)
    atlas_cable_a.add_argument("--repository-root", required=True)
    atlas_cable_a.add_argument("--dataset-root", required=True)
    atlas_cable_a.add_argument("--output-root", required=True)
    atlas_cable_a.add_argument("--output-relpath", required=True)
    atlas_cable_a.add_argument("--device")
    atlas_cable_a.set_defaults(handler=_run_diag_atlas_cable_a)
    atlas_extra = subparsers.add_parser("diag-atlas-extra-a")
    atlas_extra.add_argument("--mvtec-root", required=True)
    atlas_extra.add_argument("--visa-root", required=True)
    atlas_extra.add_argument("--output-root", required=True)
    atlas_extra.add_argument("--output-relpath", required=True)
    atlas_extra.add_argument("--device")
    atlas_extra.set_defaults(handler=_run_diag_atlas_extra_a)
    atlas_cad3 = subparsers.add_parser("diag-atlas-cad3-a")
    atlas_cad3.add_argument("--cad3-root", required=True)
    atlas_cad3.add_argument("--output-root", required=True)
    atlas_cad3.add_argument("--output-relpath", required=True)
    atlas_cad3.add_argument("--device")
    atlas_cad3.set_defaults(handler=_run_diag_atlas_cad3_a)
    a2_support = subparsers.add_parser("diag-a2-support")
    a2_support.add_argument("--paths", required=True)
    a2_support.add_argument("--a-experiment", required=True)
    a2_support.add_argument("--output", required=True)
    a2_support.add_argument("--repository-root")
    a2_support.add_argument("--device")
    a2_support.set_defaults(handler=_run_diag_a2_support)
    h_corr3 = subparsers.add_parser("diag-h-corr3")
    h_corr3.add_argument("--paths", required=True)
    h_corr3.add_argument("--a-experiment", required=True)
    h_corr3.add_argument("--output", required=True)
    h_corr3.add_argument("--repository-root")
    h_corr3.add_argument("--device")
    h_corr3.set_defaults(handler=_run_diag_h_corr3)
    h_corr3_extra = subparsers.add_parser("diag-h-corr3-extra")
    h_corr3_extra.add_argument("--paths", required=True)
    h_corr3_extra.add_argument("--visa-root", required=True)
    h_corr3_extra.add_argument("--cad3-root", required=True)
    h_corr3_extra.add_argument("--output", required=True)
    h_corr3_extra.add_argument("--repository-root")
    h_corr3_extra.add_argument("--device")
    h_corr3_extra.set_defaults(handler=_run_diag_h_corr3_extra)
    h_pose_sample = subparsers.add_parser("diag-h-pose-sample")
    h_pose_sample.add_argument("--paths", required=True)
    h_pose_sample.add_argument("--a-experiment", required=True)
    h_pose_sample.add_argument("--output", required=True)
    h_pose_sample.add_argument("--repository-root")
    h_pose_sample.add_argument("--device")
    h_pose_sample.set_defaults(handler=_run_diag_h_pose_sample)
    h_pose_sample_extra = subparsers.add_parser("diag-h-pose-sample-extra")
    h_pose_sample_extra.add_argument("--paths", required=True)
    h_pose_sample_extra.add_argument("--visa-root", required=True)
    h_pose_sample_extra.add_argument("--cad3-root", required=True)
    h_pose_sample_extra.add_argument("--output", required=True)
    h_pose_sample_extra.add_argument("--repository-root")
    h_pose_sample_extra.add_argument("--device")
    h_pose_sample_extra.set_defaults(handler=_run_diag_h_pose_sample_extra)
    a2_support_hard = subparsers.add_parser("diag-a2-support-hard")
    a2_support_hard.add_argument("--mvtec-root", required=True)
    a2_support_hard.add_argument("--visa-root", required=True)
    a2_support_hard.add_argument("--cad3-root", required=True)
    a2_support_hard.add_argument("--can-root", required=True)
    a2_support_hard.add_argument("--output", required=True)
    a2_support_hard.add_argument("--device")
    a2_support_hard.set_defaults(handler=_run_diag_a2_support_hard)
    wave1 = subparsers.add_parser("diag-wave1-label-geom")
    wave1.add_argument("--paths", required=True)
    wave1.add_argument("--a-experiment", required=True)
    wave1.add_argument("--cable-dump", required=True)
    wave1.add_argument("--can-dump", required=True)
    wave1.add_argument("--can-layout", required=True)
    wave1.add_argument("--visa-root", required=True)
    wave1.add_argument("--visa-dump-root")
    wave1.add_argument("--candle-dump")
    wave1.add_argument("--capsules-dump")
    wave1.add_argument("--cashew-dump")
    wave1.add_argument("--chewinggum-dump")
    wave1.add_argument("--output", required=True)
    wave1.add_argument("--repository-root")
    wave1.set_defaults(handler=_run_diag_wave1_label_geom)
    wave2 = subparsers.add_parser("diag-wave2-density-size")
    wave2.add_argument("--paths", required=True)
    wave2.add_argument("--a-experiment", required=True)
    wave2.add_argument("--cable-dump", required=True)
    wave2.add_argument("--can-dump", required=True)
    wave2.add_argument("--can-layout", required=True)
    wave2.add_argument("--visa-root", required=True)
    wave2.add_argument("--visa-dump-root")
    wave2.add_argument("--candle-dump")
    wave2.add_argument("--capsules-dump")
    wave2.add_argument("--cashew-dump")
    wave2.add_argument("--chewinggum-dump")
    wave2.add_argument("--wave1-summary", required=True)
    wave2.add_argument("--output", required=True)
    wave2.add_argument("--repository-root")
    wave2.set_defaults(handler=_run_diag_wave2_density_size)
    loc_overlay = subparsers.add_parser("diag-loc-pred-gt-overlay")
    loc_overlay.add_argument("--eval-root", required=True)
    loc_overlay.add_argument("--mvtec-path", required=True)
    loc_overlay.add_argument("--eval-kit", required=True)
    loc_overlay.add_argument("--output", required=True)
    loc_overlay.add_argument("--device")
    loc_overlay.set_defaults(handler=_run_diag_loc_pred_gt_overlay)
    loc_decomp = subparsers.add_parser("diag-loc-score-decomp")
    loc_decomp.add_argument("--eval-root", required=True)
    loc_decomp.add_argument("--mvtec-path", required=True)
    loc_decomp.add_argument("--eval-kit", required=True)
    loc_decomp.add_argument("--output", required=True)
    loc_decomp.add_argument("--device")
    loc_decomp.set_defaults(handler=_run_diag_loc_score_decomp)
    seas_ap = subparsers.add_parser("diag-seas-pixel-ap-buckets")
    seas_ap.add_argument("--eval-root", required=True)
    seas_ap.add_argument("--eval-kit", required=True)
    seas_ap.add_argument("--output", required=True)
    seas_ap.add_argument("--device")
    seas_ap.set_defaults(handler=_run_diag_seas_pixel_ap_buckets)
    oneshot = subparsers.add_parser("diag-oneshot-appearance")
    oneshot.add_argument("--runs-root", required=True)
    oneshot.add_argument("--output", required=True)
    oneshot.add_argument("--device")
    oneshot.set_defaults(handler=_run_diag_oneshot_appearance)
    capsules_combined = subparsers.add_parser("diag-capsules-combined-panel")
    capsules_combined.add_argument("--eval-root", required=True)
    capsules_combined.add_argument("--output", required=True)
    capsules_combined.set_defaults(handler=_run_diag_capsules_combined_panel)
    visa_type_panel = subparsers.add_parser("diag-visa-type-gt-mfinal")
    visa_type_panel.add_argument("--eval-root", required=True)
    visa_type_panel.add_argument("--output", required=True)
    visa_type_panel.add_argument("--category", required=True)
    visa_type_panel.add_argument("--type-slug", required=True, nargs="+")
    visa_type_panel.add_argument("--n-rows-per-type", type=int, default=4)
    visa_type_panel.add_argument("--eval-name", default="20260922-seas-500")
    visa_type_panel.add_argument("--layout-name", default="mvtec_layout_seas_20260922")
    visa_type_panel.set_defaults(handler=_run_diag_visa_type_gt_mfinal)
    capsules_upper = subparsers.add_parser("diag-capsules-combined-upper-area")
    capsules_upper.add_argument("--eval-root", required=True)
    capsules_upper.add_argument("--output", required=True)
    capsules_upper.add_argument("--source-root")
    capsules_upper.add_argument("--device")
    capsules_upper.set_defaults(handler=_run_diag_capsules_combined_upper_area)
    capsules_two = subparsers.add_parser("diag-capsules-combined-two-stamp")
    capsules_two.add_argument("--eval-root", required=True)
    capsules_two.add_argument("--output", required=True)
    capsules_two.add_argument("--source-root")
    capsules_two.add_argument("--device")
    capsules_two.set_defaults(handler=_run_diag_capsules_combined_two_stamp)
    tokencut_normals = subparsers.add_parser("diag-tokencut-normals")
    tokencut_normals.add_argument("--mvtec-root", required=True)
    tokencut_normals.add_argument("--visa-root", required=True)
    tokencut_normals.add_argument("--cable-experiment", required=True)
    tokencut_normals.add_argument("--output", required=True)
    tokencut_normals.add_argument("--repository-root")
    tokencut_normals.add_argument("--device")
    tokencut_normals.set_defaults(handler=_run_diag_tokencut_normals)
    tokencut_relarea = subparsers.add_parser("diag-tokencut-relarea")
    tokencut_relarea.add_argument("--identities", required=True)
    tokencut_relarea.add_argument("--mvtec-root", required=True)
    tokencut_relarea.add_argument("--visa-root", required=True)
    tokencut_relarea.add_argument("--cable-experiment", required=True)
    tokencut_relarea.add_argument("--output", required=True)
    tokencut_relarea.add_argument("--repository-root")
    tokencut_relarea.add_argument("--device")
    tokencut_relarea.add_argument("--tau", type=float, default=0.2)
    tokencut_relarea.set_defaults(handler=_run_diag_tokencut_relarea)
    h_scale_clip = subparsers.add_parser("diag-h-scale-clip")
    h_scale_clip.add_argument("--mvtec-root", required=True)
    h_scale_clip.add_argument("--visa-root", required=True)
    h_scale_clip.add_argument("--cable-experiment", required=True)
    h_scale_clip.add_argument("--output", required=True)
    h_scale_clip.add_argument("--repository-root")
    h_scale_clip.add_argument("--device")
    h_scale_clip.set_defaults(handler=_run_diag_h_scale_clip)
    proposal_cc = subparsers.add_parser("diag-proposal-tokencut-cc")
    proposal_cc.add_argument("--mvtec-root", required=True)
    proposal_cc.add_argument("--visa-root", required=True)
    proposal_cc.add_argument("--cable-experiment", required=True)
    proposal_cc.add_argument("--output", required=True)
    proposal_cc.add_argument("--repository-root")
    proposal_cc.add_argument("--device")
    proposal_cc.set_defaults(handler=_run_diag_proposal_tokencut_cc)
    site_scale = subparsers.add_parser("diag-h-site-scale-coverage")
    site_scale.add_argument("--mvtec-root", required=True)
    site_scale.add_argument("--visa-root", required=True)
    site_scale.add_argument("--cable-experiment", required=True)
    site_scale.add_argument("--output", required=True)
    site_scale.add_argument("--repository-root")
    site_scale.add_argument("--device")
    site_scale.add_argument("--n-draws", type=int, default=100)
    site_scale.set_defaults(handler=_run_diag_h_site_scale_coverage)
    quantile_cover = subparsers.add_parser("diag-a2-quantile-cover")
    quantile_cover.add_argument("--mvtec-root", required=True)
    quantile_cover.add_argument("--visa-root", required=True)
    quantile_cover.add_argument("--cable-experiment", required=True)
    quantile_cover.add_argument("--output", required=True)
    quantile_cover.add_argument("--repository-root")
    quantile_cover.add_argument("--device")
    quantile_cover.set_defaults(handler=_run_diag_a2_quantile_cover)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--repository-root", required=True)
    generate.add_argument("--dataset-root", required=True)
    generate.add_argument("--output-root", required=True)
    generate.add_argument("--output-relpath", required=True)
    generate.add_argument("--device")
    generate.set_defaults(handler=_run_official_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
