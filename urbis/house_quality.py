"""Promotion rules for learned house polygons.

An image model must beat the fixed, untouched Accra and Kampala gate before it
may replace the source-backed building footprints shown by the application.
"""
import json
from pathlib import Path


REQUIRED_CITIES = ("accra", "kampala")
MIN_F1 = 0.60
# SHA-256 over each sorted tile's filename and JPG/PNG/JSON bytes.  This binds
# promotion to the immutable 274-tile local gate, not a newly rebuilt dataset
# that happens to use the same two city names.
FIXED_GATE_SHA256 = "376a76b2a99aa0e2059c4d1c06f5e771c65464528609d99ce60a3b31eb56d7e8"


def accepted(report, min_f1=MIN_F1):
    """Return whether an evaluation report proves the production requirement."""
    try:
        if report.get("passed_app_gate") is not True:
            return False
        if float(report["min_f1"]) < min_f1:
            return False
        if len(report["checkpoint_sha256"]) != 64:
            return False
        int(report["checkpoint_sha256"], 16)
        if report["test_set_sha256"] != FIXED_GATE_SHA256:
            return False
        cities = report["cities"]
        return all(float(cities[name]["validation_selected_rule"]["f1"]) >= min_f1
                   for name in REQUIRED_CITIES)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def load_accepted(path, min_f1=MIN_F1):
    """Read *path* and return its report only when it passes the gate."""
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return report if accepted(report, min_f1) else None


def selected_houses_path(directory):
    """Choose approved model output, otherwise the source footprint baseline."""
    directory = Path(directory)
    ai = directory / "houses_ai.geojson"
    quality = directory / "houses_ai.quality.json"
    return ai if ai.is_file() and load_accepted(quality) is not None else directory / "houses.geojson"
