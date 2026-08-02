from __future__ import annotations

from pathlib import Path

from scripts.export_active_router_predictions import export_predictions


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_cheap_screen_export_is_complete(tmp_path: Path) -> None:
    result = export_predictions(
        project_root=ROOT,
        protocol_path=ROOT / "artifacts/active_router_protocol.json",
        manifest_path=ROOT / "artifacts/active_router_task_manifest.json",
        records_path=ROOT / "outputs/active_router_v1/episodes.jsonl",
        exclusions_path=ROOT / "outputs/active_router_v1/screen_exclusions.json",
        stage="cheap_medium_screen",
        output_dir=tmp_path,
    )
    assert set(result["models"]) == {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }
    assert result["models"]["openai/gpt-oss-20b"][
        "structurally_valid_count"
    ] == 10
    assert result["models"]["openai/gpt-oss-120b"][
        "structurally_valid_count"
    ] == 9
    assert all(
        row["expected_task_count"] == 12 for row in result["models"].values()
    )
