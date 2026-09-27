# contextflow/scorers/heuristic_scorer.py
"""
Defines HeuristicPriorityScorer.

A deterministic, rule-based priority scorer that evaluates messages based on:
1. Protected status (absolute override -> 100 score, critical classification)
2. Role weighting (system > user > assistant > tool)
3. Recency (position within the conversation trajectory)
4. Explicit importance levels (CRITICAL, IMPORTANT, NORMAL, DISCARDABLE)
5. Content signals (requirements, constraints, technical blocks vs. debug logs and pleasantries)
"""

import re
from typing import Dict, List, Optional

from ..message import Message, ImportanceLevel
from .base import PriorityScorer, PriorityScore


# Content pattern matchers
_HIGH_IMPORTANCE_WORDS = frozenset({
    "requirement", "requirements", "must", "always", "never",
    "critical", "crucial", "decision", "decisions", "architecture",
    "rule", "rules", "security", "config", "configuration",
    "constraint", "constraints", "password", "token", "secret",
    "credential", "credentials", "compliance", "hipaa", "gdpr",
    "do not", "cannot", "essential",
})

_LOW_IMPORTANCE_WORDS = frozenset({
    "ping", "pong", "healthcheck", "heartbeat", "debug", "trace",
    "garbage collection", "latency", "status 200", "200 ok",
})

_PLEASANTRIES = frozenset({
    "ok", "okay", "thanks", "thank you", "got it", "noted",
    "understood", "sure", "acknowledged", "hello", "hi", "hey",
    "bye", "goodbye", "sounds good", "great", "cool",
})

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_JSON_LIKE_RE  = re.compile(r"\{[\s\S]*?:[\s\S]*?\}")


class HeuristicPriorityScorer(PriorityScorer):
    """Deterministic, rule-based priority scorer.

    Evaluates messages without network calls or LLM dependencies.
    Produces scores in the range [0.0, 100.0] and classifies each message as
    `CRITICAL`, `IMPORTANT`, `NORMAL`, or `DISCARDABLE`.

    Args:
        critical_threshold: Minimum score for CRITICAL classification (default: 80.0).
        important_threshold: Minimum score for IMPORTANT classification (default: 55.0).
        normal_threshold: Minimum score for NORMAL classification (default: 25.0).
        max_recency_points: Maximum points awarded for conversational recency (default: 20.0).
        role_weights: Optional custom points per role (default: system=25, user=15, assistant=10, tool=5).
    """

    DEFAULT_ROLE_WEIGHTS: Dict[str, float] = {
        "system":    25.0,
        "user":      15.0,
        "assistant": 10.0,
        "tool":       5.0,
    }

    DEFAULT_IMPORTANCE_WEIGHTS: Dict[ImportanceLevel, float] = {
        ImportanceLevel.CRITICAL:    50.0,
        ImportanceLevel.IMPORTANT:   25.0,
        ImportanceLevel.NORMAL:      10.0,
        ImportanceLevel.DISCARDABLE: -25.0,
    }

    def __init__(
        self,
        critical_threshold:  float = 80.0,
        important_threshold: float = 55.0,
        normal_threshold:    float = 25.0,
        max_recency_points:  float = 20.0,
        role_weights:        Optional[Dict[str, float]] = None,
    ) -> None:
        if critical_threshold <= important_threshold:
            raise ValueError("critical_threshold must be greater than important_threshold.")
        if important_threshold <= normal_threshold:
            raise ValueError("important_threshold must be greater than normal_threshold.")

        self.critical_threshold:  float = float(critical_threshold)
        self.important_threshold: float = float(important_threshold)
        self.normal_threshold:    float = float(normal_threshold)
        self.max_recency_points:  float = float(max_recency_points)
        self.role_weights:        Dict[str, float] = (
            dict(role_weights) if role_weights is not None else dict(self.DEFAULT_ROLE_WEIGHTS)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        message: Message,
        index: int = 0,
        total_messages: int = 1,
    ) -> PriorityScore:
        """Score a single message using heuristic factors."""
        # 1. Protected override: Protected messages are always 100.0 / CRITICAL
        if message.protected:
            return PriorityScore(
                score=100.0,
                classification=ImportanceLevel.CRITICAL,
                message=message,
                factors={"protected": 100.0},
                reason="Message is marked as protected (never removed or summarized).",
            )

        factors: Dict[str, float] = {}
        reasons: List[str] = []

        # 2. Base points
        base_points = 10.0
        factors["base"] = base_points

        # 3. Role factor
        role_points = self.role_weights.get(message.role, 10.0)
        factors["role"] = role_points
        reasons.append(f"role={message.role} (+{role_points})")

        # 4. Recency factor
        total = max(1, total_messages)
        recency_ratio = (index + 1) / total
        recency_points = round(self.max_recency_points * recency_ratio, 1)
        factors["recency"] = recency_points
        reasons.append(f"recency={recency_points}/{self.max_recency_points}")

        # 5. Explicit importance factor
        imp_points = self.DEFAULT_IMPORTANCE_WEIGHTS.get(message.importance, 10.0)
        factors["importance"] = imp_points
        if message.importance != ImportanceLevel.NORMAL:
            reasons.append(f"importance={message.importance.value} ({imp_points:+})")

        # 6. Content signals
        content_points, content_reasons = self._analyze_content(message.content)
        factors["content"] = content_points
        reasons.extend(content_reasons)

        # Compute raw total
        raw_total = base_points + role_points + recency_points + imp_points + content_points
        final_score = max(0.0, min(100.0, round(raw_total, 1)))

        # Determine classification
        classification: ImportanceLevel
        if message.importance == ImportanceLevel.CRITICAL:
            final_score = max(final_score, 90.0)
            classification = ImportanceLevel.CRITICAL
            reasons.append("explicit critical override")
        elif message.importance == ImportanceLevel.DISCARDABLE and not message.protected:
            final_score = min(final_score, 25.0)
            classification = ImportanceLevel.DISCARDABLE
            reasons.append("explicit discardable override")
        else:
            if final_score >= self.critical_threshold:
                classification = ImportanceLevel.CRITICAL
            elif final_score >= self.important_threshold:
                classification = ImportanceLevel.IMPORTANT
            elif final_score >= self.normal_threshold:
                classification = ImportanceLevel.NORMAL
            else:
                classification = ImportanceLevel.DISCARDABLE

        return PriorityScore(
            score=final_score,
            classification=classification,
            message=message,
            factors=factors,
            reason="; ".join(reasons),
        )

    # ------------------------------------------------------------------
    # Content analysis helper
    # ------------------------------------------------------------------

    def _analyze_content(self, text: str) -> tuple[float, List[str]]:
        points = 0.0
        notes: List[str] = []
        lower = text.lower().strip()

        # Check for code blocks or structured data
        if _CODE_BLOCK_RE.search(text) or _JSON_LIKE_RE.search(text):
            points += 10.0
            notes.append("contains code/structured data (+10)")

        # Check for high-importance keywords
        words = set(re.findall(r"\b\w+\b", lower))
        high_matches = words & _HIGH_IMPORTANCE_WORDS
        if high_matches:
            term_points = min(30.0, 15.0 + 5.0 * (len(high_matches) - 1))
            points += term_points
            sample = sorted(list(high_matches))[:3]
            notes.append(f"key terms: {', '.join(sample)} (+{term_points:.0f})")

            # High density directive bonus
            if {"critical", "security", "never", "must"} & high_matches and len(high_matches) >= 3:
                points += 10.0
                notes.append("high-density critical directives (+10)")

        # Check for low-importance / ephemeral keywords
        low_matches = words & _LOW_IMPORTANCE_WORDS
        if low_matches:
            points -= 25.0
            notes.append("ephemeral/debug terms (-25)")

        # Check for pleasantries / trivial acknowledgments
        cleaned_words = [w for w in re.findall(r"\b\w+\b", lower)]
        if cleaned_words and all(w in _PLEASANTRIES for w in cleaned_words) and len(cleaned_words) <= 4:
            points -= 25.0
            notes.append("trivial pleasantry/ack (-25)")
        elif len(lower) < 10 and not high_matches:
            points -= 10.0
            notes.append("very short turn (-10)")

        return points, notes
