from dataclasses import dataclass
from datetime import datetime, timezone

from idea_pipeline.core.models import Candidate, CandidateArchetype


@dataclass
class ScoredCandidate:
    candidate: Candidate
    raw: int
    freshness: float
    fatigue: float
    final: float


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def freshness_multiplier(candidate_created_at: datetime, now: datetime) -> float:
    age_hours = (_ensure_aware(now) - _ensure_aware(candidate_created_at)).total_seconds() / 3600
    if age_hours < 20:
        return 1.0
    elif age_hours < 44:
        return 0.95
    else:
        return 0.90


def archetype_fatigue_multiplier(
    archetype: CandidateArchetype,
    recently_published: list[CandidateArchetype],
) -> float:
    penalties = {0: 0.6, 1: 0.8, 2: 0.9}
    strongest = 1.0
    for i, published_archetype in enumerate(recently_published[:3]):
        if published_archetype == archetype:
            strongest = min(strongest, penalties[i])
    return strongest


def score_candidate(
    candidate: Candidate,
    now: datetime,
    recently_published_archetypes: list[CandidateArchetype],
) -> ScoredCandidate:
    raw = candidate.score
    freshness = freshness_multiplier(candidate.created_at, now)
    fatigue = archetype_fatigue_multiplier(
        candidate.archetype, recently_published_archetypes
    )
    return ScoredCandidate(
        candidate=candidate,
        raw=raw,
        freshness=freshness,
        fatigue=fatigue,
        final=raw * freshness * fatigue,
    )


def select_best_candidate(
    candidates: list[Candidate],
    now: datetime,
    recently_published_archetypes: list[CandidateArchetype],
) -> tuple[ScoredCandidate, list[ScoredCandidate]] | None:
    """Returns (winner, all_scored_sorted) or None if no candidates."""
    if not candidates:
        return None
    scored = [
        score_candidate(c, now, recently_published_archetypes)
        for c in candidates
    ]
    scored.sort(key=lambda s: s.final, reverse=True)
    return scored[0], scored


def print_selection_ranking(scored: list[ScoredCandidate], limit: int = 20) -> None:
    print(f"  Candidate ranking ({min(limit, len(scored))} of {len(scored)}):")
    for i, s in enumerate(scored[:limit], 1):
        marker = " *" if i == 1 else ""
        print(
            f"    {i:>2}. [{s.candidate.archetype.value:<15}] "
            f"{s.candidate.theme[:50]:<50}  "
            f"raw={s.raw:>3}  fresh={s.freshness:.2f}  "
            f"fatigue={s.fatigue:.2f}  final={s.final:.1f}{marker}"
        )
