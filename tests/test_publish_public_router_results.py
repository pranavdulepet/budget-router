from pathlib import Path

from scripts.publish_public_router_results import publish


ROOT = Path(__file__).parents[1]


def test_publish_public_router_results_is_sanitized_and_consistent(
    tmp_path: Path,
) -> None:
    report = publish(
        ROOT / "outputs/public_router_v1/study/results.json",
        ROOT / "outputs/public_router_v1/open_source/evaluation.json",
        ROOT / "outputs/public_router_v1/arc_agi/results.json",
        ROOT / "outputs/active_router_v1/cheap_medium_screen_gate.json",
        ROOT / "outputs/active_router_v1/higher_cost_screen_gate.json",
        ROOT / "outputs/active_router_v1/collection_summary.json",
        tmp_path / "public.json",
    )

    assert report["inferential_summary"]["confirmatory_primary"] == (
        "failed_on_id_test"
    )
    assert report["deployable_artifact"]["tests"]["id_test"][
        "cost_saving_fraction"
    ] > 0.30
    assert report["arc_agi_sequential_replication"]["quality_difference"] < -0.20
    assert (tmp_path / "public.json").is_file()
