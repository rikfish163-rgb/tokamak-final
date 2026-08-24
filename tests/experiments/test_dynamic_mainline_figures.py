from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib as mpl
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = (
    REPO_ROOT
    / "experiments"
    / "transfer_methods_2026"
    / "paper_grade_package"
    / "scripts"
)


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        spec.loader.exec_module(loaded)
    finally:
        sys.path.pop(0)
    return loaded


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def test_publication_style_has_six_point_floor_and_redundant_series_encoding() -> None:
    style = _load_module("dynamic_publication_style", "publication_style.py")

    style.apply_nature_style()
    configured_fonts = [
        mpl.rcParams[key]
        for key in (
            "font.size",
            "axes.labelsize",
            "axes.titlesize",
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
        )
    ]
    encodings = [style.series_encoding(index) for index in range(6)]

    assert min(configured_fonts) >= 6.0
    assert style.minimum_configured_font_size() >= 6.0
    assert all(item["color"] in style.OKABE_ITO for item in encodings)
    assert len({item["marker"] for item in encodings}) == len(encodings)
    assert len({item["linestyle"] for item in encodings}) == len(encodings)


def test_empty_roots_render_five_fail_closed_figures_and_portable_manifest(
    tmp_path: Path,
) -> None:
    figures = _load_module(
        "generate_dynamic_mainline_figures",
        "generate_dynamic_mainline_figures.py",
    )
    east_root = tmp_path / "east"
    jtext_root = tmp_path / "jtext"
    output_dir = tmp_path / "figures"
    east_root.mkdir()
    jtext_root.mkdir()

    manifest = figures.generate_dynamic_mainline_figures(
        east_root=east_root,
        jtext_root=jtext_root,
        output_dir=output_dir,
    )

    manifest_path = output_dir / "figure_manifest_v2.json"
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["schema_version"] == "figure-manifest-v2"
    assert manifest["style"]["minimum_font_size_pt"] >= 6.0
    assert manifest["style"]["palette"] == "Okabe-Ito"
    assert manifest["style"]["grayscale_redundant"] is True
    assert [figure["id"] for figure in manifest["figures"]] == [
        "F1", "F2", "F3", "F4", "F5",
    ]
    assert str(tmp_path) not in json.dumps(manifest, sort_keys=True)

    f1 = manifest["figures"][0]
    assert f1["kind"] == "conceptual_non_data"
    event_panel = next(panel for panel in f1["panels"] if panel["id"] == "F1C")
    assert event_panel["event_order"] == [
        "predict",
        "endpoint_seal",
        "optional_intra_update",
        "later_endpoint_prediction",
        "shot_seal",
        "query",
        "reveal",
        "persistent_update_or_rollback",
    ]

    for figure in manifest["figures"]:
        assert set(figure["outputs"]) == {"pdf", "svg", "png"}
        for extension, output in figure["outputs"].items():
            path = output_dir / output["path"]
            assert path.is_file() and path.stat().st_size > 0
            assert _is_sha256(output["sha256"])
            if extension == "pdf":
                assert path.read_bytes().startswith(b"%PDF-")
            elif extension == "svg":
                assert ET.parse(path).getroot().tag.endswith("svg")
            else:
                with Image.open(path) as image:
                    dpi = image.info.get("dpi")
                    assert dpi is not None
                    assert min(dpi) >= 599.0

    required_panel_fields = {
        "statistic",
        "uncertainty",
        "cohort",
        "physical_shot_count",
        "model_seed_count",
        "seed_count",
        "claim_status",
        "inputs",
        "input_sha256",
        "row_selector",
        "parent_sha256",
        "label_policy",
        "label_policy_sha256",
        "generator_sha256",
        "config_sha256",
        "caption_sha256",
        "output_sha256",
    }
    for figure in manifest["figures"][1:]:
        for panel in figure["panels"]:
            assert panel["kind"] in {"quantitative", "unavailable"}
            assert panel["status"] == "unavailable"
            assert required_panel_fields <= panel.keys()
            assert panel["reason"]
            assert panel["values"] == []
            assert panel["input_sha256"]
            assert all(_is_sha256(value) for value in panel["input_sha256"])
            assert all(_is_sha256(value) for value in panel["output_sha256"].values())
