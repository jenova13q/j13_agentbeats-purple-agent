import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text

from messenger import Messenger

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency may be absent locally
    OpenAI = None


SYSTEM_PROMPT = """You are a negotiation agent for an item-allocation game.

You must respond with JSON only.

When proposing, return:
{"allocation_self": [int, ...], "allocation_other": [int, ...]}

Rules:
- allocation_self[i] + allocation_other[i] must equal quantities[i]
- all values must be non-negative integers
- prefer deals where your own value stays above BATNA
- start firm, then make small concessions if needed
- keep higher-value items for yourself when possible
"""


@dataclass
class NegotiationState:
    offers_to_other_by_game: dict[int, list[list[int]]] = field(default_factory=dict)


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.state = NegotiationState()
        self.model = os.environ.get("MODEL") or "gpt-4.1-mini"
        self.client = self._init_client()

    def _init_client(self):
        if OpenAI is None:
            return None

        openai_key = os.environ.get("OPENAI_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")

        if openai_key:
            return OpenAI(api_key=openai_key)
        if openrouter_key:
            return OpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
            )
        return None

    def _parse_observation(self, message_text: str) -> dict[str, Any]:
        patterns = [
            r"```json\s*(.*?)```",
            r"```\s*(.*?)```",
            r"Observation:\s*(\{.*?\})",
        ]

        for pattern in patterns:
            match = re.search(pattern, message_text, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        brace_start = message_text.find("{")
        if brace_start != -1:
            depth = 0
            for i, char in enumerate(message_text[brace_start:], brace_start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(message_text[brace_start : i + 1])
                        except json.JSONDecodeError:
                            break

        return {}

    def _extract_json_from_text(self, text: str) -> dict[str, Any] | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        brace_start = text.find("{")
        if brace_start == -1:
            return None

        depth = 0
        for i, char in enumerate(text[brace_start:], brace_start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break

        return None

    def _calculate_value(self, allocation: list[int], valuations: list[int]) -> int:
        return sum(a * v for a, v in zip(allocation, valuations))

    def _other_from_self(self, allocation_self: list[int], quantities: list[int]) -> list[int]:
        return [q - a for q, a in zip(quantities, allocation_self)]

    def _self_from_other(self, allocation_other: list[int], quantities: list[int]) -> list[int]:
        return [q - a for q, a in zip(quantities, allocation_other)]

    def _compute_rule_based_offer(
        self,
        quantities: list[int],
        valuations_self: list[int],
        batna_self: int,
        round_index: int,
        max_rounds: int,
    ) -> tuple[list[int], list[int]] | None:
        item_order = sorted(range(len(valuations_self)), key=lambda i: valuations_self[i], reverse=True)
        allocation_self = [0] * len(quantities)
        allocation_other = list(quantities)

        progress = round_index / max(max_rounds, 1)
        if progress < 0.34:
            target_ratio = 0.82
        elif progress < 0.67:
            target_ratio = 0.72
        else:
            target_ratio = 0.62

        max_value = self._calculate_value(quantities, valuations_self)
        target_value = max(batna_self, int(max_value * target_ratio))

        for index in item_order:
            while allocation_other[index] > 0 and self._calculate_value(allocation_self, valuations_self) < target_value:
                allocation_other[index] -= 1
                allocation_self[index] += 1

        if self._calculate_value(allocation_self, valuations_self) < batna_self:
            return None

        if all(x == 0 for x in allocation_self) or all(x == 0 for x in allocation_other):
            for index in reversed(item_order):
                if allocation_self[index] > 1:
                    allocation_self[index] -= 1
                    allocation_other[index] += 1
                    break

        if self._calculate_value(allocation_self, valuations_self) < batna_self:
            return None

        return allocation_self, allocation_other

    def _prepare_context(self, obs: dict[str, Any]) -> str:
        quantities = obs.get("quantities", [])
        valuations_self = obs.get("valuations_self", [])
        batna_self = obs.get("batna_self", 0)
        round_index = obs.get("round_index", 1)
        max_rounds = obs.get("max_rounds", 5)
        game_index = obs.get("game_index", 0)

        history = self.state.offers_to_other_by_game.get(game_index, [])
        history_text = ""
        if history:
            history_text = "\nPrevious offers to opponent:\n"
            for idx, offer in enumerate(history[-3:], 1):
                history_text += f"- Offer {idx}: allocation_other={offer}\n"

        return f"""Game state:
- quantities: {quantities}
- my valuations: {valuations_self}
- my BATNA: {batna_self}
- round: {round_index} of {max_rounds}
{history_text}

Return one valid JSON proposal only.
Prefer keeping more of the higher-valued items.
Do not return explanations.
"""

    def _call_llm_for_offer(self, obs: dict[str, Any]) -> dict[str, Any] | None:
        if self.client is None:
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._prepare_context(obs)},
                ],
            )
            content = response.choices[0].message.content or ""
            return self._extract_json_from_text(content.strip())
        except Exception:
            return None

    def _validate_offer(
        self, proposal: dict[str, Any], quantities: list[int], valuations_self: list[int], batna_self: int
    ) -> tuple[list[int] | None, list[int] | None]:
        allocation_self = proposal.get("allocation_self")
        allocation_other = proposal.get("allocation_other")

        if allocation_self is None:
            return None, None

        allocation_self = [int(x) for x in allocation_self]
        if allocation_other is None:
            allocation_other = self._other_from_self(allocation_self, quantities)
        else:
            allocation_other = [int(x) for x in allocation_other]

        if len(allocation_self) != len(quantities) or len(allocation_other) != len(quantities):
            return None, None

        for i in range(len(quantities)):
            if allocation_self[i] < 0 or allocation_other[i] < 0:
                return None, None
            if allocation_self[i] + allocation_other[i] != quantities[i]:
                return None, None

        if self._calculate_value(allocation_self, valuations_self) < batna_self:
            return None, None

        return allocation_self, allocation_other

    def _accept_or_walk(self, obs: dict[str, Any]) -> dict[str, Any]:
        pending = obs.get("pending_offer")
        if not pending:
            return {"action": "WALK", "reason": "no_pending_offer"}

        valuations_self = obs.get("valuations_self", [])
        batna_self = obs.get("batna_self", 0)
        round_index = obs.get("round_index", 1)
        max_rounds = obs.get("max_rounds", 5)
        discount = obs.get("discount", 1.0)
        offered_self = pending.get("offer_allocation_self", [])

        if not offered_self:
            return {"action": "WALK", "reason": "bad_pending_offer"}

        value = self._calculate_value(offered_self, valuations_self)
        discounted_batna = batna_self * (discount ** max(round_index - 1, 0))
        progress = round_index / max(max_rounds, 1)

        if progress < 0.34:
            threshold = discounted_batna
        elif progress < 0.67:
            threshold = discounted_batna * 0.8
        else:
            threshold = discounted_batna * 0.6

        if value >= threshold:
            return {"action": "ACCEPT"}
        return {"action": "WALK"}

    def _propose(self, obs: dict[str, Any]) -> dict[str, Any]:
        quantities = obs.get("quantities", [])
        valuations_self = obs.get("valuations_self", [])
        batna_self = obs.get("batna_self", 0)
        round_index = obs.get("round_index", 1)
        max_rounds = obs.get("max_rounds", 5)
        game_index = obs.get("game_index", 0)

        llm_result = self._call_llm_for_offer(obs)
        if llm_result is not None:
            allocation_self, allocation_other = self._validate_offer(
                llm_result, quantities, valuations_self, batna_self
            )
            if allocation_self is not None and allocation_other is not None:
                self.state.offers_to_other_by_game.setdefault(game_index, []).append(allocation_other)
                return {
                    "allocation_self": allocation_self,
                    "allocation_other": allocation_other,
                }

        fallback = self._compute_rule_based_offer(
            quantities, valuations_self, batna_self, round_index, max_rounds
        )
        if fallback is None:
            return {"action": "WALK", "reason": "no_valid_offer"}

        allocation_self, allocation_other = fallback
        self.state.offers_to_other_by_game.setdefault(game_index, []).append(allocation_other)
        return {
            "allocation_self": allocation_self,
            "allocation_other": allocation_other,
        }

    def _handle_negotiation_message(self, message_text: str) -> dict[str, Any]:
        obs = self._parse_observation(message_text)
        if not obs:
            return {"action": "WALK", "reason": "parse_error"}

        action = str(obs.get("action", "")).upper()
        if action == "ACCEPT_OR_REJECT":
            return self._accept_or_walk(obs)
        return self._propose(obs)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message) or ""
        response = self._handle_negotiation_message(input_text)
        response_text = json.dumps(response)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_text))],
            name="NegotiationResponse",
        )
