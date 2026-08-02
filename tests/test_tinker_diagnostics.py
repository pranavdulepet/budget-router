from decimal import Decimal
from unittest import TestCase

from budget_router.mini_swe_tinker import TinkerMiniSweModel
from budget_router.pricing import ModelPrice
from budget_router.providers import ModelResponse
from budget_router.providers.tinker import bounded_tinker_seed
from budget_router.tinker_diagnostics import render_tinker_prompt
from budget_router.types import TokenUsage


class EncodedTextChunk:
    pass


class FakePrompt:
    def __init__(self, tokens: list[int]) -> None:
        self._tokens = tokens
        self.chunks = [EncodedTextChunk()]

    def to_ints(self) -> list[int]:
        return list(self._tokens)


class FakeRenderer:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.tools = []
        self.messages = []
        self.effort = None

    def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
        self.tools = tools
        self.system_prompt = system_prompt
        return [{"role": "system", "content": f"tools:{len(tools)}"}]

    def build_generation_prompt(self, messages, effort=None):
        self.messages = messages
        self.effort = effort
        return FakePrompt([1, 2, 3, 4])


class TinkerPromptDiagnosticTests(TestCase):
    def test_rendering_matches_adapter_system_and_tool_handling(self) -> None:
        renderer = FakeRenderer()
        rendered = render_tinker_prompt(
            renderer,
            [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "task"},
            ],
            [{"name": "bash"}],
            effort=0.9,
        )

        self.assertEqual(renderer.system_prompt, "policy")
        self.assertEqual(renderer.tools, [{"name": "bash"}])
        self.assertEqual(
            [message["role"] for message in rendered.rendered_messages],
            ["system", "user"],
        )
        self.assertEqual(renderer.effort, 0.9)
        self.assertEqual(rendered.token_ids, (1, 2, 3, 4))

    def test_diagnostics_are_structural_and_context_aware(self) -> None:
        rendered = render_tinker_prompt(
            FakeRenderer(),
            [{"role": "user", "content": "not included in diagnostics"}],
            [],
        )
        diagnostics = rendered.diagnostics(
            context_tokens=8,
            max_output_tokens=4,
            stop_sequences=[99],
        )

        self.assertEqual(diagnostics["input_tokens"], 4)
        self.assertEqual(diagnostics["context_utilization"], 1.0)
        self.assertTrue(diagnostics["within_context"])
        self.assertTrue(diagnostics["text_only"])
        self.assertEqual(diagnostics["chunk_types"], ["EncodedTextChunk"])
        self.assertEqual(diagnostics["stop_sequences"], [99])
        self.assertNotIn("not included", str(diagnostics))

    def test_tinker_seed_is_bounded_to_signed_32_bits(self) -> None:
        self.assertIsNone(bounded_tinker_seed(None))
        self.assertEqual(bounded_tinker_seed(20_260_726), 20_260_726)
        self.assertEqual(
            bounded_tinker_seed(20_260_726_000),
            20_260_726_000 % (2**31),
        )

    def test_mini_swe_call_seed_is_not_multiplied(self) -> None:
        class Adapter:
            seeds = []

            async def _generate_from_messages(self, request, messages):
                del messages
                self.seeds.append(request.seed)
                return ModelResponse(
                    model=request.model,
                    content="",
                    usage=TokenUsage(input_tokens=4, output_tokens=1),
                )

        adapter = Adapter()
        model = TinkerMiniSweModel(
            adapter,
            model_name="model",
            renderer="renderer",
            reasoning="enabled",
            temperature=0.7,
            seed=20_260_726,
            hard_limit_usd=Decimal("1"),
            price=ModelPrice(
                model="model",
                input_per_million_usd=Decimal("1"),
                output_per_million_usd=Decimal("1"),
                cached_input_per_million_usd=Decimal("1"),
            ),
            price_snapshot="prices",
            max_output_tokens=16,
        )

        model.query([{"role": "user", "content": "task"}])
        model.query([{"role": "user", "content": "task"}])

        self.assertEqual(adapter.seeds, [20_260_726, 20_260_727])
