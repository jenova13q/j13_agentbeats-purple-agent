import re
from dataclasses import dataclass, field

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


@dataclass
class NegotiationState:
    round_number: int = 0
    last_offer_for_us: int | None = None
    opponent_style: str = "unknown"
    accepted: bool = False
    history: list[str] = field(default_factory=list)


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.state = NegotiationState()
        self.offer_sequence = [70, 67, 64, 61, 58, 55]

    def _detect_opponent_style(self, text: str) -> str:
        lowered = text.lower()
        soft_markers = ["fair", "compromise", "balance", "mutual", "together", "both"]
        hard_markers = ["final", "take it or leave it", "must", "non-negotiable", "ultimatum"]
        manipulative_markers = ["only reasonable", "obviously", "clearly", "everyone knows", "be fair"]

        if any(marker in lowered for marker in hard_markers):
            return "hard"
        if any(marker in lowered for marker in manipulative_markers):
            return "manipulative"
        if any(marker in lowered for marker in soft_markers):
            return "soft"
        return "unknown"

    def _extract_offer_for_us(self, text: str) -> int | None:
        lowered = text.lower()
        patterns = [
            r"you get (\d{1,3})",
            r"you take (\d{1,3})",
            r"your share(?: is| =)? (\d{1,3})",
            r"you receive (\d{1,3})",
            r"split (?:it )?(\d{1,3})\/(\d{1,3})",
            r"(\d{1,3})\s*-\s*(\d{1,3}) split",
            r"(\d{1,3})\/(\d{1,3}) split",
        ]

        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            if len(match.groups()) == 1:
                value = int(match.group(1))
                if 0 <= value <= 100:
                    return value
            if len(match.groups()) == 2:
                left = int(match.group(1))
                right = int(match.group(2))
                total = left + right
                if total == 100:
                    return max(left, right)

        opponent_keeps = re.search(r"i (?:get|keep|take) (\d{1,3})", lowered)
        if opponent_keeps:
            their_share = int(opponent_keeps.group(1))
            if 0 <= their_share <= 100:
                return 100 - their_share

        if "equal split" in lowered or "split evenly" in lowered or "50/50" in lowered:
            return 50

        return None

    def _next_target_share(self) -> int:
        idx = min(self.state.round_number, len(self.offer_sequence) - 1)
        target = self.offer_sequence[idx]

        if self.state.opponent_style == "soft":
            return max(target - 2, 55)
        if self.state.opponent_style in {"hard", "manipulative"}:
            return min(target + 2, 70)
        return target

    def _should_accept(self, offer_for_us: int | None) -> bool:
        if offer_for_us is None:
            return False
        if offer_for_us >= 60:
            return True
        if offer_for_us >= 55 and self.state.round_number >= 4:
            return True
        return False

    def _build_response(self, input_text: str) -> str:
        self.state.round_number += 1
        self.state.history.append(input_text.strip())
        self.state.opponent_style = self._detect_opponent_style(input_text)
        self.state.last_offer_for_us = self._extract_offer_for_us(input_text)

        lowered = input_text.lower()

        if any(token in lowered for token in ["agreed", "deal", "accepted", "i accept"]) and self.state.round_number > 1:
            self.state.accepted = True
            return "Agreed. We have a deal. I accept the current terms and am ready to finalize."

        if self._should_accept(self.state.last_offer_for_us):
            self.state.accepted = True
            return (
                f"That works for me. I accept a {self.state.last_offer_for_us}/{100 - self.state.last_offer_for_us} split "
                "and consider the negotiation settled."
            )

        target_share = self._next_target_share()

        if self.state.round_number == 1:
            return (
                "Let's aim for an efficient agreement. "
                f"My opening proposal is a {target_share}/{100 - target_share} split in my favor. "
                "I can make small reciprocal concessions if you improve your offer."
            )

        if self.state.opponent_style == "manipulative":
            return (
                "I want a mutually beneficial deal, but appeals to fairness alone are not enough. "
                f"My current position is {target_share}/{100 - target_share} in my favor. "
                "If you want movement from me, make a concrete reciprocal concession."
            )

        if self.state.opponent_style == "hard":
            return (
                "Your position is too rigid for me to accept as is. "
                f"My counteroffer is {target_share}/{100 - target_share} in my favor. "
                "If you can improve the terms, I can continue negotiating."
            )

        if self.state.last_offer_for_us is not None and self.state.last_offer_for_us < 50:
            return (
                "That proposal leaves me with too little value to accept. "
                f"A workable counteroffer is {target_share}/{100 - target_share} in my favor. "
                "I am open to a balanced compromise, but not to a one-sided deal."
            )

        return (
            "I am still open to agreement. "
            f"My current best offer is {target_share}/{100 - target_share} in my favor. "
            "If you move closer to that range, we can likely close this negotiation."
        )

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        response_text = self._build_response(input_text or "")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Reviewing the current negotiation state."),
        )
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_text))],
            name="NegotiationResponse",
        )
