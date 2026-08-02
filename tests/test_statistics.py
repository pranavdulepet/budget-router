import tempfile
from decimal import Decimal
from pathlib import Path
from unittest import TestCase

from budget_router.bandit import (
    OuterAction,
    OuterBanditSessionAdapter,
    RetrievalMemory,
    TerminalOnlyBandit,
)
from budget_router.baselines import CheapestModelRouter
from budget_router.calibration import (
    IsotonicCalibrator,
    MonotoneBudgetCalibrator,
    ipcw_weights,
)
from budget_router.estimators import ActionValueEstimator, CompetingRiskBudgetEstimator
from budget_router.censoring import censoring_stress_test
from budget_router.drift import apply_price_shock, cumulative_regret, remove_model
from budget_router.experiments import (
    CheckpointCandidate,
    PilotGate,
    PilotModelResult,
    TaskRecord,
    allocate_credits,
    derive_budget_levels,
    effective_sample_size,
    offline_policy_coverage_gate,
    repository_split,
    select_checkpoint_counterfactuals,
)
from budget_router.metrics import (
    brier_score,
    false_feasible_rate,
    integrated_brier_score,
    normalized_success_auc,
    repository_bootstrap_interval,
)
from budget_router.pilot import (
    ToolProtocolEpisode,
    replacement_for,
    summarize_tool_protocol_pilot,
)
from budget_router.tournament import PolicyEvaluation, select_reference_router
from budget_router.types import (
    GoalContext,
    Operation,
    RouterAction,
    RouterState,
    TerminalOutcome,
)

from tests.helpers import make_prices


class CalibrationTests(TestCase):
    def test_isotonic_predictions_are_monotone(self) -> None:
        calibrator = IsotonicCalibrator().fit(
            [0.1, 0.2, 0.3, 0.4], [False, True, False, True]
        )
        predictions = calibrator.predict([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(predictions, sorted(predictions))

    def test_budget_calibrator_is_monotone(self) -> None:
        calibrator = MonotoneBudgetCalibrator().fit(
            [1, 2, 3, 4], [False, True, False, True]
        )
        self.assertEqual(calibrator.predict([1, 2, 3, 4]), sorted(calibrator.probabilities))

    def test_ipcw_zeroes_censored_rows(self) -> None:
        self.assertEqual(ipcw_weights([0.5, 0.5], [False, True]), [2.0, 0.0])

    def test_competing_risk_cumulative_incidence(self) -> None:
        estimator = CompetingRiskBudgetEstimator().fit(
            [1, 2, 3],
            ["success", "failure", "censored"],
            bins=[1, 2, 3],
        )
        success, failure = estimator.predict(3)
        self.assertGreater(success, 0)
        self.assertGreater(failure, 0)
        self.assertLessEqual(success + failure, 1)

    def test_action_value_smoothing(self) -> None:
        estimator = ActionValueEstimator().fit(
            ["a", "a", "b"], [True, False, True], [1.0, 2.0, 3.0]
        )
        probability, cost = estimator.predict("a")
        self.assertTrue(0 < probability < 1)
        self.assertGreater(cost, 0)


class ExperimentTests(TestCase):
    def test_repository_split_has_no_overlap(self) -> None:
        tasks = [
            TaskRecord(f"{repo}-{index}", repo)
            for repo in ("a", "b", "c", "d", "e")
            for index in range(4)
        ]
        manifest = repository_split(tasks, seed=1)
        membership = {
            task: split
            for split, identifiers in (
                ("train", manifest.train),
                ("calibration", manifest.calibration),
                ("test", manifest.test),
            )
            for task in identifiers
        }
        for repo, split in manifest.repositories.items():
            self.assertEqual(
                {membership[task.task_id] for task in tasks if task.repository == repo},
                {split},
            )

    def test_manifest_freeze_is_immutable(self) -> None:
        tasks = [TaskRecord("a", "ra"), TaskRecord("b", "rb"), TaskRecord("c", "rc")]
        manifest = repository_split(tasks)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            manifest.freeze(path)
            manifest.freeze(path)
            path.write_text('{"manifest_hash":"wrong"}', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                manifest.freeze(path)

    def test_training_cost_quantiles_round_up(self) -> None:
        budgets = derive_budget_levels(["0.001", "0.019", "0.025", "0.10"])
        self.assertEqual(budgets[0], Decimal("0.01"))
        self.assertEqual(budgets[-1], Decimal("0.10"))
        self.assertEqual(tuple(budgets), tuple(sorted(budgets)))

    def test_pilot_gate_requires_structure_and_credit(self) -> None:
        result = PilotModelResult("m", 12, 10, Decimal("1"), 2.0, 0.1)
        gate = PilotGate((result,), Decimal("10"), Decimal("10"), Decimal("9"))
        self.assertFalse(gate.may_collect)
        with self.assertRaises(PermissionError):
            gate.require_collection_approval()

    def test_checkpoint_selection_respects_budget(self) -> None:
        candidates = [
            CheckpointCandidate("a", "uncertain", 1, 1, 1, Decimal("0.6")),
            CheckpointCandidate("b", "early", 0.5, 0.5, 0.5, Decimal("0.4")),
        ]
        selected = select_checkpoint_counterfactuals(
            candidates, budget_usd=Decimal("0.6")
        )
        self.assertLessEqual(
            sum((item.estimated_cost_usd for item in selected), Decimal("0")),
            Decimal("0.6"),
        )

    def test_credit_allocations_sum(self) -> None:
        allocation = allocate_credits(Decimal("100"))
        self.assertEqual(
            allocation["checkpoint_acquisition"]
            + allocation["router_training"]
            + allocation["on_policy_validation_test"],
            Decimal("100"),
        )

    def test_synthetic_tool_protocol_summary(self) -> None:
        episodes = [
            ToolProtocolEpisode(
                "m",
                f"task-{index}",
                True,
                True,
                True,
                True,
                True,
                Decimal("0.1"),
                10,
            )
            for index in range(12)
        ]
        result = summarize_tool_protocol_pilot(episodes)[0]
        self.assertTrue(result.passes)
        self.assertEqual(result.throughput_tasks_per_hour, 360)
        self.assertEqual(replacement_for("bad", ()), "Qwen/Qwen3.5-4B")

    def test_offline_policy_effective_sample_gate(self) -> None:
        self.assertEqual(effective_sample_size([1, 1, 1]), 3)
        self.assertTrue(
            offline_policy_coverage_gate(
                [1, 1, 1], minimum_effective_sample_size=3
            )
        )


class MetricAndBanditTests(TestCase):
    def test_metrics(self) -> None:
        self.assertAlmostEqual(normalized_success_auc([1, 2, 3], [0, 0.5, 1]), 0.5)
        self.assertAlmostEqual(brier_score([0.2, 0.8], [False, True]), 0.04)
        self.assertAlmostEqual(
            integrated_brier_score(
                [[0.2, 0.8], [0.1, 0.9]],
                [[False, True], [False, True]],
            ),
            0.025,
        )
        self.assertEqual(false_feasible_rate([0.9, 0.7], [False, True]), 1.0)
        low, high = repository_bootstrap_interval(
            [0, 1, 0, 1], ["a", "a", "b", "b"], samples=100, seed=1
        )
        self.assertLessEqual(low, high)

    def test_terminal_bandit_updates(self) -> None:
        actions = [OuterAction("a"), OuterAction("b", "probe")]
        bandit = TerminalOnlyBandit(actions, ["bias"], seed=1)
        chosen = bandit.select({"bias": 1})
        bandit.update_terminal(
            {"bias": 1},
            chosen,
            TerminalOutcome(True, True, True),
        )
        self.assertIn(chosen.key, bandit.scores({"bias": 1}))
        with self.assertRaises(TypeError):
            bandit.update_terminal({"bias": 1}, chosen, object())  # type: ignore[arg-type]

    def test_session_bandit_adapter_uses_terminal_feedback(self) -> None:
        bandit = TerminalOnlyBandit([OuterAction("a")], ["bias"])
        adapter = OuterBanditSessionAdapter(
            bandit, lambda goal, state: {"bias": 1.0}
        )
        before = bandit.scores({"bias": 1})["a:direct"]
        adapter.update_terminal(
            GoalContext("fix", "repo", ("a",)),
            RouterState(current_model="a"),
            RouterAction(Operation.STOP),
            TerminalOutcome(True, True, True),
        )
        after = bandit.scores({"bias": 1})["a:direct"]
        self.assertNotEqual(before, after)

    def test_retrieval_memory(self) -> None:
        actions = [OuterAction("a"), OuterAction("b")]
        memory = RetrievalMemory(k=1)
        memory.add([0.0], "a:direct", 0)
        memory.add([1.0], "b:direct", 1)
        self.assertEqual(memory.select([0.9], actions).model, "b")

    def test_drift_price_shock_and_removal(self) -> None:
        prices = make_prices()
        shocked = apply_price_shock(prices, "cheap")
        self.assertEqual(
            shocked.price_for("cheap").input_per_million_usd,
            Decimal("2"),
        )
        removed = remove_model(prices, "strong")
        self.assertNotIn("strong", removed.models)
        self.assertEqual(cumulative_regret([0, 1], [1, 1]), (1.0, 1.0))

    def test_cheapest_control(self) -> None:
        prices = make_prices()
        goal = GoalContext("fix", "repo", ("cheap", "strong"))
        selected = CheapestModelRouter().select(
            goal, budget_usd=Decimal("1"), prices=prices
        )
        self.assertEqual(selected, "cheap")

    def test_censoring_stress(self) -> None:
        result = censoring_stress_test(
            [0.1, 0.3, 0.7, 0.9],
            [False, False, True, True],
            probability=0.25,
            outcome_odds_ratio=2,
            seed=1,
        )
        self.assertEqual(result.mechanism, "outcome_dependent")
        self.assertEqual(result.retained + result.censored, 4)


class TournamentTests(TestCase):
    @staticmethod
    def evaluation(name: str, auc: float, *, fixed: bool = False) -> PolicyEvaluation:
        return PolicyEvaluation(
            name,
            auc,
            integrated_brier=0.1,
            false_feasible_rate=0.1,
            hard_cap_violations=0,
            auc_interval=(auc - 0.01, auc + 0.01),
            routing_latency_ms=1,
            routing_cost_usd=0,
            complexity_rank=0 if fixed else 1,
            is_fixed_baseline=fixed,
        )

    def test_fixed_baseline_wins_negative_result(self) -> None:
        result = select_reference_router(
            [self.evaluation("fixed", 0.7, fixed=True), self.evaluation("learned", 0.69)],
            budget_only_integrated_brier=0.2,
        )
        self.assertEqual(result.selected.name, "fixed")
        self.assertTrue(result.negative_result)

    def test_hard_cap_violator_is_excluded(self) -> None:
        fixed = self.evaluation("fixed", 0.5, fixed=True)
        invalid = PolicyEvaluation(
            "invalid", 0.9, 0.1, 0.1, 1, (0.8, 1.0), 1, 0, 1
        )
        result = select_reference_router(
            [fixed, invalid], budget_only_integrated_brier=0.2
        )
        self.assertEqual(result.excluded["invalid"], "hard_cap_violation")
