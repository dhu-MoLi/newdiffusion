#!/usr/bin/env python3
"""Stage-0 A2 entrance diagnostic.

For every category / defect type / reconstruction backend it measures:

* fill consistency: feature cosine and colour L1 between the filled hole and
  the surrounding normal ring;
* proposal material consistency: proposal/support ratio, similarity gap, tau;
* location plausibility: SDF and radial position of the real defect, the
  proposal and the TokenCut site in the object frame.

No diffusion training and no generated images are required.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.config import load_config
from cadiff.v06_a.structure import physical_sdf_px_512
from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v06_c.visa_pilot import (
    fewshot_identity_records,
    load_visa_identity,
    load_visa_inventory_for_split,
    normal_identity_records,
    slug_type,
)
from cadiff.v07.location import ObjectLocationPrior
from cadiff.v07.proposal_probe import (
    SiteResult,
    fill_consistency_from_features,
    location_stats,
    probe_from_features,
    site_probe,
)
from cadiff.v07.reconstruction import make_reconstructor, ring_mask
from cadiff.v07.site_selection import proposal_sites, select_site

DEFAULTS = {
    "paths": "configs/paths/asus-mvtec.yaml",
    "split": "mvtec",
    "categories": ["bottle", "carpet", "grid", "transistor"],
    "backends": ["neighbor", "telea", "ring"],
    "normals_per_type": 2,
    "sources_per_type": 1,
    "max_probes_per_type": 8,
    "ring_radius": 24,
    "quantile": 0.60,
    "atlas_references": 8,
    "site_method": "support_cc",
    "site_temperature": 1.0,
    "site_similarity_weight": 0.3,
    "device": "cuda:0",
    "seed": 20261110,
}


class FeatureCache:
    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self._features: dict[str, Any] = {}
        self._supports: dict[str, np.ndarray] = {}
        self._sdfs: dict[str, np.ndarray] = {}

    def features(self, rgb: np.ndarray, key: str) -> Any:
        if key not in self._features:
            self._features[key] = self.encoder.encode(rgb)
        return self._features[key]

    def support(self, rgb: np.ndarray, valid: np.ndarray, key: str) -> np.ndarray:
        if key not in self._supports:
            features = self.features(rgb, key)
            self._supports[key] = tokencut_object_mask(features.tokens, valid)
        return self._supports[key]

    def sdf(self, support: np.ndarray, key: str) -> np.ndarray:
        if key not in self._sdfs:
            self._sdfs[key] = physical_sdf_px_512(support)
        return self._sdfs[key]


def _load_category(
    project: Any,
    category: str,
    split: str,
    *,
    max_normals: int,
    max_sources: int,
):
    inventory = load_visa_inventory_for_split(
        project.dataset_root, category, split, None
    )
    sources: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for record in fewshot_identity_records(inventory):
        if counts[record.defect_type] >= max_sources:
            continue
        loaded = load_visa_identity(record, dataset_root=inventory.source_root)
        sources[record.defect_type].append((record.item_id, loaded))
        counts[record.defect_type] += 1
    normals: list[tuple[str, Any]] = []
    for record in normal_identity_records(inventory)[:max_normals]:
        loaded = load_visa_identity(record, dataset_root=inventory.source_root)
        normals.append((record.item_id, loaded))
    return inventory, sources, normals


def _select_site(
    *,
    method: str,
    target_support: np.ndarray,
    target_valid: np.ndarray,
    proposal: np.ndarray,
    similarity: np.ndarray,
    seed: int,
    prior: ObjectLocationPrior | None,
    defect_type: str,
    temperature: float,
    similarity_weight: float,
) -> tuple[SiteResult, float]:
    if method == "support_cc":
        result = site_probe(
            target_support=target_support, proposal=proposal, seed=seed
        )
        return result, float("nan")
    sites = proposal_sites(
        proposal,
        target_support,
        target_valid,
        similarity=similarity,
        min_pixels=64,
    )
    if not sites:
        return (
            SiteResult(
                found=False,
                mask=None,
                pixel_count=0,
                site_index=-1,
                site_count=0,
            ),
            float("nan"),
        )
    prior_scores = None
    prior_score = float("nan")
    if method == "proposal_cc_prior":
        if prior is None:
            raise ValueError("proposal_cc_prior requires a location prior")
        prior_scores = [
            prior.score_site(site, target_support, defect_type) for site in sites
        ]
    site = select_site(
        sites,
        seed=seed,
        similarity_weight=similarity_weight,
        prior_scores=prior_scores,
        temperature=temperature,
    )
    if prior is not None:
        prior_score = prior.score_mask(site.mask, target_support, defect_type)
    return (
        SiteResult(
            found=True,
            mask=site.mask,
            pixel_count=int(site.pixels),
            site_index=int(site.index),
            site_count=len(sites),
        ),
        prior_score,
    )


def _make_row(
    *,
    category: str,
    defect_type: str,
    backend: str,
    prototype_mode: str,
    source_id: str,
    target_id: str,
    probe: Any,
    fill: dict[str, float],
    real: dict[str, float],
    proposal_location: dict[str, float],
    site: Any,
    site_location: dict[str, float],
    site_prior_score: float = float("nan"),
) -> dict[str, Any]:
    return {
        "category": category,
        "defect_type": defect_type,
        "type_slug": slug_type(defect_type),
        "backend": backend,
        "prototype_mode": prototype_mode,
        "source_id": source_id,
        "target_id": target_id,
        "fill_feature_cos": fill.get("fill_feature_cos", float("nan")),
        "fill_color_l1": fill.get("fill_color_l1", float("nan")),
        "tau": probe.stats["tau"],
        "proposal_support_ratio": probe.stats["proposal_support_ratio"],
        "proposal_components": probe.stats["proposal_components"],
        "support_valid_ratio": probe.stats["support_valid_ratio"],
        "proposal_mean_similarity": probe.stats["proposal_mean_similarity"],
        "similarity_gap": probe.stats["similarity_gap"],
        "real_sdf_median": real["sdf_median"],
        "real_radial_median": real["radial_median"],
        "real_mask_pixels": real["mask_pixels"],
        "proposal_sdf_median": proposal_location["sdf_median"],
        "proposal_radial_median": proposal_location["radial_median"],
        "proposal_mask_pixels": proposal_location["mask_pixels"],
        "site_found": 1.0 if site.found else 0.0,
        "site_pixels": float(site.pixel_count),
        "site_count": float(site.site_count),
        "site_support_ratio": float(
            site.pixel_count / max(proposal_location["support_pixels"], 1.0)
        ),
        "site_proposal_ratio": float(
            site.pixel_count / max(proposal_location["mask_pixels"], 1.0)
        ),
        "site_sdf_median": site_location["sdf_median"],
        "site_radial_median": site_location["radial_median"],
        "site_prior_score": site_prior_score,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    payload: dict[str, Any] = {"n": len(rows)}
    for key in numeric:
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), (int, float))
            and np.isfinite(float(row[key]))
        ]
        if values:
            payload[f"{key}_mean"] = float(np.mean(values))
            payload[f"{key}_std"] = float(np.std(values))
            payload[f"{key}_count"] = len(values)
    return payload


def _markdown(
    *,
    title: str,
    overall: dict[str, dict[str, Any]],
    per_type: dict[str, dict[str, Any]],
) -> str:
    keys = [
        "fill_feature_cos",
        "fill_color_l1",
        "proposal_support_ratio",
        "proposal_components",
        "proposal_mean_similarity",
        "similarity_gap",
        "real_sdf_median",
        "proposal_sdf_median",
        "site_sdf_median",
        "real_radial_median",
        "proposal_radial_median",
        "site_radial_median",
        "site_support_ratio",
        "site_proposal_ratio",
        "site_count",
        "site_prior_score",
        "site_pixels",
    ]
    lines = [f"# {title}", "", "## Overall", ""]
    lines.append("| backend | n | " + " | ".join(keys) + " |")
    lines.append("|" + "---|" * (len(keys) + 2))
    for name, payload in sorted(overall.items()):
        cells = []
        for key in keys:
            value = payload.get(f"{key}_mean")
            cells.append("n/a" if value is None else f"{value:.4f}")
        lines.append(f"| {name} | {payload['n']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Per defect type", ""])
    lines.append("| group | n | " + " | ".join(keys[:8]) + " |")
    lines.append("|" + "---|" * (len(keys[:8]) + 2))
    for name, payload in sorted(per_type.items()):
        cells = []
        for key in keys[:8]:
            value = payload.get(f"{key}_mean")
            cells.append("n/a" if value is None else f"{value:.4f}")
        lines.append(f"| {name} | {payload['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default=DEFAULTS["paths"])
    parser.add_argument("--split", default=DEFAULTS["split"])
    parser.add_argument("--categories", nargs="+", default=DEFAULTS["categories"])
    parser.add_argument("--backends", nargs="+", default=DEFAULTS["backends"])
    parser.add_argument("--normals-per-type", type=int, default=DEFAULTS["normals_per_type"])
    parser.add_argument("--sources-per-type", type=int, default=DEFAULTS["sources_per_type"])
    parser.add_argument("--max-probes-per-type", type=int, default=DEFAULTS["max_probes_per_type"])
    parser.add_argument("--ring-radius", type=int, default=DEFAULTS["ring_radius"])
    parser.add_argument("--quantile", type=float, default=DEFAULTS["quantile"])
    parser.add_argument("--atlas-references", type=int, default=DEFAULTS["atlas_references"])
    parser.add_argument(
        "--site-method",
        choices=["support_cc", "proposal_cc", "proposal_cc_prior"],
        default=DEFAULTS["site_method"],
    )
    parser.add_argument("--site-temperature", type=float, default=DEFAULTS["site_temperature"])
    parser.add_argument(
        "--site-similarity-weight",
        type=float,
        default=DEFAULTS["site_similarity_weight"],
    )
    parser.add_argument("--device", default=DEFAULTS["device"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sd2-root", type=Path, default=None)
    parser.add_argument("--learned-root", type=Path, default=None)
    arguments = parser.parse_args(argv)

    project = load_config(ROOT / arguments.paths)
    name = arguments.name or time.strftime("diag_%Y%m%d_%H%M%S")
    output = (
        arguments.output
        if arguments.output is not None
        else Path(project.runs_root) / "v07" / "diag_proposal" / name
    )
    output.mkdir(parents=True, exist_ok=False)

    encoder = load_official_dinov2_encoder(device=arguments.device)
    cache = FeatureCache(encoder)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for category in arguments.categories:
        inventory, sources, normals = _load_category(
            project,
            category,
            arguments.split,
            max_normals=max(arguments.atlas_references, arguments.normals_per_type),
            max_sources=arguments.sources_per_type,
        )
        if not normals:
            raise RuntimeError(f"{category} has no normal targets")
        reference_rgb = tuple(item.rgb for _, item in normals[: arguments.atlas_references])
        reference_valid = tuple(
            item.valid for _, item in normals[: arguments.atlas_references]
        )
        learned_checkpoint = None
        if "learned" in arguments.backends:
            if arguments.learned_root is None:
                raise ValueError("--learned-root is required for the learned backend")
            candidates = sorted(
                (arguments.learned_root / category).rglob("checkpoint.pt")
            )
            if not candidates:
                raise FileNotFoundError(
                    f"no learned checkpoint under {arguments.learned_root / category}"
                )
            learned_checkpoint = candidates[-1]
            print(f"[{category}] learned checkpoint: {learned_checkpoint}", flush=True)
        location_prior = None
        if arguments.site_method == "proposal_cc_prior":
            location_prior = ObjectLocationPrior.from_inventory(inventory, encoder)
            print(
                f"[{category}] location prior {location_prior.describe()}",
                flush=True,
            )
        for backend in arguments.backends:
            prototype_mode = "ring" if backend == "ring" else "hole"
            reconstructor = None
            if backend != "ring":
                reconstructor = make_reconstructor(
                    backend,
                    references=reference_rgb,
                    reference_valids=reference_valid,
                    sd2_root=arguments.sd2_root,
                    learned_checkpoint=learned_checkpoint,
                    device=arguments.device,
                    prompt=f"a photo of a normal {category}",
                )
            for defect_type, items in sorted(sources.items()):
                probes = 0
                for source_index, (source_id, source) in enumerate(
                    items[: arguments.sources_per_type]
                ):
                    for target_offset in range(arguments.normals_per_type):
                        if probes >= arguments.max_probes_per_type:
                            break
                        target_id, target = normals[
                            (source_index * arguments.normals_per_type + target_offset)
                            % len(normals)
                        ]
                        seed = arguments.seed + len(rows) + target_offset
                        try:
                            if prototype_mode == "ring":
                                source_key = f"source:{source_id}:ring"
                                source_features = cache.features(source.rgb, source_key)
                                filled = None
                                source_mask = ring_mask(
                                    source.anomaly_gt,
                                    source.valid,
                                    radius=arguments.ring_radius,
                                )
                            else:
                                source_key = f"source:{source_id}:{backend}"
                                filled = reconstructor(
                                    source.rgb, source.anomaly_gt, source.valid
                                )
                                source_features = cache.features(filled, source_key)
                                source_mask = source.anomaly_gt
                            source_support = cache.support(
                                filled if filled is not None else source.rgb,
                                source.valid,
                                source_key,
                            )
                            target_key = f"normal:{target_id}"
                            target_features = cache.features(target.rgb, target_key)
                            target_support = cache.support(
                                target.rgb, target.valid, target_key
                            )
                            probe = probe_from_features(
                                source_features=source_features,
                                source_mask=source_mask,
                                source_valid=source.valid,
                                source_support=source_support,
                                target_features=target_features,
                                target_valid=target.valid,
                                target_support=target_support,
                                quantile=arguments.quantile,
                            )
                            if filled is not None:
                                fill = fill_consistency_from_features(
                                    source_rgb=source.rgb,
                                    filled_rgb=filled,
                                    hole=source.anomaly_gt,
                                    valid=source.valid,
                                    features=source_features,
                                    ring_radius=arguments.ring_radius,
                                )
                            else:
                                fill = {
                                    "fill_feature_cos": float("nan"),
                                    "fill_color_l1": float("nan"),
                                }
                            real = location_stats(
                                mask=source.anomaly_gt,
                                support=source_support,
                                sdf=cache.sdf(source_support, source_key),
                            )
                            proposal_location = location_stats(
                                mask=probe.proposal,
                                support=target_support,
                                sdf=cache.sdf(target_support, target_key),
                            )
                            site, site_prior_score = _select_site(
                                method=arguments.site_method,
                                target_support=target_support,
                                target_valid=target.valid,
                                proposal=probe.proposal,
                                similarity=probe.similarity,
                                seed=seed,
                                prior=location_prior,
                                defect_type=defect_type,
                                temperature=arguments.site_temperature,
                                similarity_weight=arguments.site_similarity_weight,
                            )
                            if site.found and site.mask is not None:
                                site_location = location_stats(
                                    mask=site.mask,
                                    support=target_support,
                                    sdf=cache.sdf(target_support, target_key),
                                )
                            else:
                                site_location = {
                                    "sdf_median": float("nan"),
                                    "radial_median": float("nan"),
                                    "mask_pixels": 0.0,
                                }
                            rows.append(
                                _make_row(
                                    category=category,
                                    defect_type=defect_type,
                                    backend=backend,
                                    prototype_mode=prototype_mode,
                                    source_id=source_id,
                                    target_id=target_id,
                                    probe=probe,
                                    fill=fill,
                                    real=real,
                                    proposal_location=proposal_location,
                                    site=site,
                                    site_location=site_location,
                                    site_prior_score=site_prior_score,
                                )
                            )
                            probes += 1
                        except Exception as error:  # keep the diagnostic running
                            failures.append(
                                {
                                    "category": category,
                                    "defect_type": defect_type,
                                    "backend": backend,
                                    "source_id": source_id,
                                    "target_id": target_id,
                                    "error": str(error),
                                }
                            )
                    if probes >= arguments.max_probes_per_type:
                        break
                print(
                    f"[{category}] {backend} {prototype_mode} {defect_type}: {probes} probes",
                    flush=True,
                )

    by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        backend = f"{row['backend']}({row['prototype_mode']})"
        by_backend[backend].append(row)
        by_type[f"{row['category']}/{row['defect_type']}/{backend}"].append(row)
    overall = {name: _aggregate(group) for name, group in by_backend.items()}
    per_type = {name: _aggregate(group) for name, group in by_type.items()}

    payload = {
        "schema": "v07_diag_proposal_v1",
        "config": {
            "paths": arguments.paths,
            "split": arguments.split,
            "categories": arguments.categories,
            "backends": arguments.backends,
            "normals_per_type": arguments.normals_per_type,
            "sources_per_type": arguments.sources_per_type,
            "max_probes_per_type": arguments.max_probes_per_type,
            "ring_radius": arguments.ring_radius,
            "quantile": arguments.quantile,
            "site_method": arguments.site_method,
            "site_temperature": arguments.site_temperature,
            "site_similarity_weight": arguments.site_similarity_weight,
            "device": arguments.device,
            "seed": arguments.seed,
        },
        "overall": overall,
        "per_type": per_type,
        "failures": failures,
        "rows": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(
        _markdown(
            title=f"Stage-0 A2 diagnostic: {name}",
            overall=overall,
            per_type=per_type,
        ),
        encoding="utf-8",
    )
    print(f"rows={len(rows)} failures={len(failures)} output={output}")
    if failures:
        print(json.dumps(failures[:5], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
