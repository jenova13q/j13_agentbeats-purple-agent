import json
import logging
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


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a negotiation agent for an item-allocation game.

You must respond with JSON only.

When proposing, prefer selecting from the provided allocation catalog.
If a catalog is present, return:
{"choice_id": <int>}

Otherwise, return:
{"allocation_self": [int, ...], "allocation_other": [int, ...]}

Primary objective:
- maximize a closeable Nash-style deal
- good deals improve your value above BATNA while still leaving the opponent with a meaningful share
- a slightly less greedy accepted deal is better than a rejected extreme deal

Strategy:
- keep the items you value most
- give up items you value least first
- start near the suggested target, then make small monotonic concessions
- in the last round, prefer a closeable deal over a greedy dead-end

Rules:
- allocation_self[i] + allocation_other[i] must equal quantities[i]
- all values must be non-negative integers
- if valid catalog options are provided, prefer the best catalog choice over inventing a new split
"""


def _uses_max_completion_tokens(model: str) -> bool:
    lowered = (model or "").lower()
    # OpenAI reasoning families and GPT-5 use max_completion_tokens instead of max_tokens.
    return (
        lowered.startswith("gpt-5")
        or lowered.startswith("gpt-5.")
        or lowered.startswith("o1")
        or lowered.startswith("o3")
        or lowered.startswith("o4")
    )


@dataclass
class NegotiationState:
    offers_to_other_by_game: dict[int, list[list[int]]] = field(default_factory=dict)
    offers_to_self_by_game: dict[int, list[list[int]]] = field(default_factory=dict)
    incoming_offers_to_self_by_game: dict[int, list[list[int]]] = field(default_factory=dict)
    best_incoming_value_by_game: dict[int, int] = field(default_factory=dict)
    best_incoming_offer_to_self_by_game: dict[int, list[int]] = field(default_factory=dict)


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.state = NegotiationState()
        self.model = os.environ.get("LLM_MODEL") or "gpt-5-mini"
        self.client = self._init_client()
        logger.warning(
            "Purple agent LLM config: client_enabled=%s model=%s openai_key=%s",
            self.client is not None,
            self.model,
            bool(os.environ.get("OPENAI_API_KEY")),
        )

    def _init_client(self):
        if OpenAI is None:
            return None

        openai_key = os.environ.get("OPENAI_API_KEY")

        if openai_key:
            return OpenAI(api_key=openai_key)
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

    def _parse_allocation_catalog(self, message_text: str) -> list[dict[str, Any]]:
        marker = "Allocation catalog"
        marker_index = message_text.find(marker)
        if marker_index == -1:
            return []

        tail = message_text[marker_index:]
        match = re.search(r"```json\s*(.*?)```", tail, re.DOTALL | re.IGNORECASE)
        if not match:
            return []

        try:
            data = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return []

        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

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

        max_value = self._calculate_value(quantities, valuations_self)
        target_value = self._target_value(max_value, batna_self, round_index, max_rounds)

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

    def _target_surplus_ratio(self, round_index: int, max_rounds: int) -> float:
        progress = round_index / max(max_rounds, 1)
        if progress <= 0.2:
            return 0.50
        if progress <= 0.4:
            return 0.44
        if progress <= 0.6:
            return 0.38
        if progress <= 0.8:
            return 0.30
        return 0.20

    def _target_value(self, max_value: int, batna_self: int, round_index: int, max_rounds: int) -> int:
        surplus = max(max_value - batna_self, 0)
        return max(batna_self, int(batna_self + surplus * self._target_surplus_ratio(round_index, max_rounds)))

    def _prepare_context(self, obs: dict[str, Any]) -> str:
        quantities = obs.get("quantities", [])
        valuations_self = obs.get("valuations_self", [])
        batna_self = obs.get("batna_self", 0)
        round_index = obs.get("round_index", 1)
        max_rounds = obs.get("max_rounds", 5)
        game_index = obs.get("game_index", 0)
        discount = obs.get("discount", 1.0)
        max_value = self._calculate_value(quantities, valuations_self)
        target_value = self._target_value(max_value, batna_self, round_index, max_rounds)
        discounted_batna = batna_self * (discount ** max(round_index - 1, 0))
        ranking = sorted(range(len(valuations_self)), key=lambda i: valuations_self[i], reverse=True)

        history = self.state.offers_to_other_by_game.get(game_index, [])
        history_text = ""
        if history:
            history_text = "\nPrevious offers to opponent:\n"
            for idx, offer in enumerate(history[-3:], 1):
                my_side = self._self_from_other(offer, quantities)
                my_value = self._calculate_value(my_side, valuations_self)
                history_text += f"- Offer {idx}: gave={offer}, kept={my_side}, my_value={my_value}\n"

        return f"""Game state:
- quantities: {quantities}
- my valuations: {valuations_self}
- my BATNA: {batna_self}
- my discounted BATNA now: {discounted_batna:.1f}
- my maximum possible value: {max_value}
- round: {round_index} of {max_rounds}
- target value this round: about {target_value}
- item priority for me: {ranking}
{history_text}

Return one valid JSON proposal only.
Aim for a closeable Nash-style deal:
- stay above BATNA
- keep more of the higher-valued items
- give up lower-valued items first
- avoid extreme all-or-nothing splits
- make small monotonic concessions across rounds
If a catalog is present, prefer a choice_id from the catalog.
Do not return explanations.
"""

    def _score_candidate(
        self,
        allocation_self: list[int],
        allocation_other: list[int],
        valuations_self: list[int],
        batna_self: int,
        round_index: int,
        max_rounds: int,
        previous_offer_to_other: list[int] | None = None,
    ) -> float:
        my_value = self._calculate_value(allocation_self, valuations_self)
        if my_value < batna_self:
            return float("-inf")

        total_value = self._calculate_value(
            [a + b for a, b in zip(allocation_self, allocation_other)],
            valuations_self,
        )
        their_proxy = max(total_value - my_value, 0)
        my_surplus = max(my_value - batna_self, 0)
        target_value = self._target_value(total_value, batna_self, round_index, max_rounds)
        closeness_penalty = abs(my_value - target_value) * 0.06
        nash_proxy = my_surplus * (their_proxy + 1)
        extreme_penalty = 0.0
        if all(x == 0 for x in allocation_self) or all(x == 0 for x in allocation_other):
            extreme_penalty += 200.0
        if sum(1 for x in allocation_other if x == 0) >= max(1, len(allocation_other) - 1):
            extreme_penalty += 60.0

        concession_penalty = 0.0
        if previous_offer_to_other is not None:
            give_back = sum(
                max(prev - current, 0) for prev, current in zip(previous_offer_to_other, allocation_other)
            )
            abrupt_concession = sum(
                max(current - prev - 1, 0) for prev, current in zip(previous_offer_to_other, allocation_other)
            )
            concession_penalty = 0.75 * give_back + 0.2 * abrupt_concession

        return nash_proxy + 0.25 * my_value - closeness_penalty - concession_penalty - extreme_penalty

    def _best_catalog_option(
        self,
        catalog: list[dict[str, Any]],
        quantities: list[int],
        valuations_self: list[int],
        batna_self: int,
        round_index: int,
        max_rounds: int,
        previous_offer_to_other: list[int] | None = None,
    ) -> dict[str, Any] | None:
        best_option = None
        best_score = float("-inf")

        for option in catalog:
            allocation_self = option.get("allocation_self")
            allocation_other = option.get("allocation_other")
            if not isinstance(allocation_self, list) or not isinstance(allocation_other, list):
                continue

            try:
                alloc_self = [int(x) for x in allocation_self]
                alloc_other = [int(x) for x in allocation_other]
            except Exception:
                continue

            if len(alloc_self) != len(quantities) or len(alloc_other) != len(quantities):
                continue
            if any(a < 0 or b < 0 for a, b in zip(alloc_self, alloc_other)):
                continue
            if any((a + b) != q for a, b, q in zip(alloc_self, alloc_other, quantities)):
                continue

            score = self._score_candidate(
                alloc_self,
                alloc_other,
                valuations_self,
                batna_self,
                round_index,
                max_rounds,
                previous_offer_to_other=previous_offer_to_other,
            )
            if score > best_score:
                best_score = score
                best_option = {
                    "choice_id": option.get("choice_id", option.get("id")),
                    "allocation_self": alloc_self,
                    "allocation_other": alloc_other,
                    "score": score,
                }

        return best_option

    def _prepare_catalog_context(
        self,
        obs: dict[str, Any],
        catalog: list[dict[str, Any]],
        top_options: list[dict[str, Any]],
    ) -> str:
        base = self._prepare_context(obs)
        option_lines = []
        for option in top_options:
            option_lines.append(
                f"- choice_id={option['choice_id']}, allocation_self={option['allocation_self']}, "
                f"allocation_other={option['allocation_other']}, score={option['score']:.2f}"
            )

        return (
            f"{base}\n"
            "Allocation catalog is available. Choose one of the valid catalog options below.\n"
            "Prefer a closeable Nash-style option: strong for you, but not so extreme that it is likely to be rejected.\n"
            "Return JSON only in the form {\"choice_id\": <int>}.\n"
            "Top candidate options:\n"
            + "\n".join(option_lines)
        )

    def _select_close_catalog_option(
        self,
        scored_options: list[dict[str, Any]],
        valuations_self: list[int],
        batna_self: int,
        round_index: int,
        max_rounds: int,
        best_incoming_value: int = 0,
    ) -> dict[str, Any] | None:
        if not scored_options:
            return None

        if round_index < max_rounds - 1:
            return None

        close_floor = max(batna_self, int(best_incoming_value * 1.02))
        best_close = None
        best_close_score = float("-inf")
        for option in scored_options:
            my_value = self._calculate_value(option["allocation_self"], valuations_self)
            if my_value < close_floor:
                continue

            their_proxy = max(
                self._calculate_value(
                    [a + b for a, b in zip(option["allocation_self"], option["allocation_other"])],
                    valuations_self,
                )
                - my_value,
                0,
            )
            target_value = self._target_value(
                self._calculate_value(
                    [a + b for a, b in zip(option["allocation_self"], option["allocation_other"])],
                    valuations_self,
                ),
                batna_self,
                round_index,
                max_rounds,
            )
            close_score = their_proxy * 0.8 - abs(my_value - target_value) * 0.04 + option["score"] * 0.2
            if close_score > best_close_score:
                best_close_score = close_score
                best_close = option

        return best_close

    def _call_llm_for_offer(
        self,
        obs: dict[str, Any],
        catalog: list[dict[str, Any]] | None = None,
        top_options: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if self.client is None:
            logger.warning("LLM disabled for PROPOSE; using fallback policy.")
            return None

        try:
            logger.warning("Calling LLM for PROPOSE with model=%s", self.model)
            user_prompt = self._prepare_context(obs)
            if catalog and top_options:
                user_prompt = self._prepare_catalog_context(obs, catalog, top_options)
            request_kwargs = {
                "model": self.model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if _uses_max_completion_tokens(self.model):
                request_kwargs["max_completion_tokens"] = 300
            else:
                request_kwargs["max_tokens"] = 300

            response = self.client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or ""
            logger.warning("LLM call succeeded for PROPOSE.")
            return self._extract_json_from_text(content.strip())
        except Exception as exc:
            logger.exception("LLM call failed for PROPOSE: %s", exc)
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
        game_index = obs.get("game_index", 0)
        quantities = obs.get("quantities", [])
        offered_self = pending.get("offer_allocation_self", [])

        if not offered_self:
            return {"action": "WALK", "reason": "bad_pending_offer"}

        self.state.incoming_offers_to_self_by_game.setdefault(game_index, []).append(list(offered_self))
        value = self._calculate_value(offered_self, valuations_self)
        best_seen = max(value, self.state.best_incoming_value_by_game.get(game_index, 0))
        self.state.best_incoming_value_by_game[game_index] = best_seen
        if value >= best_seen:
            self.state.best_incoming_offer_to_self_by_game[game_index] = list(offered_self)
        discounted_batna = batna_self * (discount ** max(round_index - 1, 0))
        max_value = self._calculate_value(quantities, valuations_self)
        target_value = self._target_value(max_value, batna_self, round_index, max_rounds)
        final_offer = self._compute_rule_based_offer(quantities, valuations_self, batna_self, max_rounds, max_rounds)
        final_offer_value = (
            self._calculate_value(final_offer[0], valuations_self) if final_offer is not None else discounted_batna
        )

        if round_index >= max_rounds:
            threshold = max(discounted_batna, min(best_seen, final_offer_value))
        elif round_index >= max_rounds - 1:
            threshold = max(discounted_batna, best_seen * 0.97, final_offer_value * 0.92, target_value * 0.62)
        elif round_index > 1:
            threshold = max(discounted_batna, target_value * 0.78)
        else:
            threshold = max(discounted_batna, target_value * 0.92)

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
        raw_message = obs.get("_raw_message", "")
        catalog = self._parse_allocation_catalog(raw_message)
        previous_offers = self.state.offers_to_other_by_game.get(game_index, [])
        previous_offer_to_other = previous_offers[-1] if previous_offers else None
        best_incoming_value = self.state.best_incoming_value_by_game.get(game_index, 0)

        best_catalog = None
        top_options: list[dict[str, Any]] = []
        close_catalog = None
        if catalog:
            scored = []
            for option in catalog:
                best = self._best_catalog_option(
                    [option],
                    quantities,
                    valuations_self,
                    batna_self,
                    round_index,
                    max_rounds,
                    previous_offer_to_other=previous_offer_to_other,
                )
                if best is not None:
                    scored.append(best)
            scored.sort(key=lambda item: item["score"], reverse=True)
            if scored:
                best_catalog = scored[0]
                top_options = scored[: min(5, len(scored))]
                close_catalog = self._select_close_catalog_option(
                    scored,
                    valuations_self,
                    batna_self,
                    round_index,
                    max_rounds,
                    best_incoming_value=best_incoming_value,
                )
                logger.warning(
                    "Catalog parsed with %s valid options; best_choice_id=%s best_score=%.2f",
                    len(scored),
                    best_catalog.get("choice_id"),
                    best_catalog.get("score", float("-inf")),
                )

        llm_result = self._call_llm_for_offer(obs, catalog=catalog, top_options=top_options)
        chosen_id = None
        if llm_result is not None:
            for key in ("choice_id", "choice_idx", "choice", "id"):
                if key in llm_result:
                    chosen_id = llm_result.get(key)
                    break

        if chosen_id is not None and top_options:
            for option in top_options:
                if option.get("choice_id") == chosen_id:
                    allocation_self = option["allocation_self"]
                    allocation_other = option["allocation_other"]
                    self.state.offers_to_other_by_game.setdefault(game_index, []).append(allocation_other)
                    self.state.offers_to_self_by_game.setdefault(game_index, []).append(allocation_self)
                    return {
                        "choice_id": chosen_id,
                        "allocation_self": allocation_self,
                        "allocation_other": allocation_other,
                    }

        if llm_result is not None:
            allocation_self, allocation_other = self._validate_offer(
                llm_result, quantities, valuations_self, batna_self
            )
            if allocation_self is not None and allocation_other is not None:
                self.state.offers_to_other_by_game.setdefault(game_index, []).append(allocation_other)
                self.state.offers_to_self_by_game.setdefault(game_index, []).append(allocation_self)
                return {
                    "allocation_self": allocation_self,
                    "allocation_other": allocation_other,
                }

        if close_catalog is not None:
            allocation_self = close_catalog["allocation_self"]
            allocation_other = close_catalog["allocation_other"]
            self.state.offers_to_other_by_game.setdefault(game_index, []).append(allocation_other)
            self.state.offers_to_self_by_game.setdefault(game_index, []).append(allocation_self)
            logger.warning("Using close-oriented catalog option with choice_id=%s", close_catalog.get("choice_id"))
            return {
                "choice_id": close_catalog.get("choice_id"),
                "allocation_self": allocation_self,
                "allocation_other": allocation_other,
            }

        if best_catalog is not None:
            allocation_self = best_catalog["allocation_self"]
            allocation_other = best_catalog["allocation_other"]
            self.state.offers_to_other_by_game.setdefault(game_index, []).append(allocation_other)
            self.state.offers_to_self_by_game.setdefault(game_index, []).append(allocation_self)
            logger.warning("Using best catalog option as fallback with choice_id=%s", best_catalog.get("choice_id"))
            return {
                "choice_id": best_catalog.get("choice_id"),
                "allocation_self": allocation_self,
                "allocation_other": allocation_other,
            }

        logger.warning("Falling back to deterministic proposal policy.")
        fallback = self._compute_rule_based_offer(
            quantities, valuations_self, batna_self, round_index, max_rounds
        )
        if fallback is None:
            return {"action": "WALK", "reason": "no_valid_offer"}

        allocation_self, allocation_other = fallback
        self.state.offers_to_other_by_game.setdefault(game_index, []).append(allocation_other)
        self.state.offers_to_self_by_game.setdefault(game_index, []).append(allocation_self)
        return {
            "allocation_self": allocation_self,
            "allocation_other": allocation_other,
        }

    def _handle_negotiation_message(self, message_text: str) -> dict[str, Any]:
        obs = self._parse_observation(message_text)
        if not obs:
            return {"action": "WALK", "reason": "parse_error"}
        obs["_raw_message"] = message_text

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
