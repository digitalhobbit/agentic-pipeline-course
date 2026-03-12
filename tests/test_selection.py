import uuid
from datetime import datetime, timedelta, timezone

import pytest

from idea_pipeline.core.models import Candidate, CandidateArchetype
from idea_pipeline.pipeline.selection import (
    archetype_fatigue_multiplier,
    freshness_multiplier,
    score_candidate,
    select_best_candidate,
    similarity_multiplier,
)


def _make_candidate(
    archetype: CandidateArchetype = CandidateArchetype.META_TREND,
    score: int = 75,
    created_at: datetime | None = None,
    **overrides,
) -> Candidate:
    defaults = {
        "id": uuid.uuid4(),
        "archetype": archetype,
        "theme": "Test Theme",
        "why_now": "Test why now",
        "score": score,
        "one_liner": "Test one-liner",
        "target_customer": "Test customer",
        "problem_to_solve": "Test problem",
        "solution_overview": "Test solution",
        "supporting_article_ids": [str(uuid.uuid4())],
        "run_id": uuid.uuid4(),
        "selected": False,
        "created_at": created_at or datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Candidate(**defaults)


class TestFreshnessMultiplier:
    def test_under_20_hours(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        created = now - timedelta(hours=10)
        assert freshness_multiplier(created, now) == 1.0

    def test_exactly_0_hours(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        assert freshness_multiplier(now, now) == 1.0

    def test_at_19_hours(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        created = now - timedelta(hours=19)
        assert freshness_multiplier(created, now) == 1.0

    def test_at_20_hours(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        created = now - timedelta(hours=20)
        assert freshness_multiplier(created, now) == 0.95

    def test_between_20_and_44_hours(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        created = now - timedelta(hours=30)
        assert freshness_multiplier(created, now) == 0.95

    def test_at_43_hours(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        created = now - timedelta(hours=43)
        assert freshness_multiplier(created, now) == 0.95

    def test_at_44_hours(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        created = now - timedelta(hours=44)
        assert freshness_multiplier(created, now) == 0.90

    def test_over_44_hours(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        created = now - timedelta(hours=60)
        assert freshness_multiplier(created, now) == 0.90


class TestArchetypeFatigueMultiplier:
    def test_no_published_history(self):
        result = archetype_fatigue_multiplier(CandidateArchetype.META_TREND, [])
        assert result == 1.0

    def test_different_archetype_no_penalty(self):
        result = archetype_fatigue_multiplier(
            CandidateArchetype.META_TREND,
            [CandidateArchetype.FRICTION_POINT],
        )
        assert result == 1.0

    def test_same_as_most_recent(self):
        result = archetype_fatigue_multiplier(
            CandidateArchetype.META_TREND,
            [CandidateArchetype.META_TREND],
        )
        assert result == 0.6

    def test_same_as_second_most_recent(self):
        result = archetype_fatigue_multiplier(
            CandidateArchetype.META_TREND,
            [CandidateArchetype.FRICTION_POINT, CandidateArchetype.META_TREND],
        )
        assert result == 0.8

    def test_same_as_third_most_recent(self):
        result = archetype_fatigue_multiplier(
            CandidateArchetype.META_TREND,
            [
                CandidateArchetype.FRICTION_POINT,
                CandidateArchetype.RABBIT_HOLE,
                CandidateArchetype.META_TREND,
            ],
        )
        assert result == 0.9

    def test_strongest_penalty_wins(self):
        """If archetype matches both most recent and 3rd most recent, use 0.6."""
        result = archetype_fatigue_multiplier(
            CandidateArchetype.META_TREND,
            [
                CandidateArchetype.META_TREND,
                CandidateArchetype.FRICTION_POINT,
                CandidateArchetype.META_TREND,
            ],
        )
        assert result == 0.6

    def test_only_considers_top_3(self):
        """4th-most-recent match should not apply any penalty."""
        result = archetype_fatigue_multiplier(
            CandidateArchetype.META_TREND,
            [
                CandidateArchetype.FRICTION_POINT,
                CandidateArchetype.RABBIT_HOLE,
                CandidateArchetype.FRICTION_POINT,
                CandidateArchetype.META_TREND,  # 4th — ignored
            ],
        )
        assert result == 1.0


class TestSimilarityMultiplier:
    def test_below_safe_zone(self):
        assert similarity_multiplier(0.0) == 1.0
        assert similarity_multiplier(0.5) == 1.0
        assert similarity_multiplier(0.69) == 1.0

    def test_at_safe_zone_boundary(self):
        assert similarity_multiplier(0.7) == pytest.approx(1.0)

    def test_linear_midpoint(self):
        # Midpoint of 0.7–0.85 is 0.775 → penalty = (0.775 - 0.7) / 0.15 = 0.5
        assert similarity_multiplier(0.775) == pytest.approx(0.5)

    def test_linear_quarter(self):
        # 0.7375 → penalty = (0.7375 - 0.7) / 0.15 = 0.25
        assert similarity_multiplier(0.7375) == pytest.approx(0.75)

    def test_just_below_kill_zone(self):
        assert similarity_multiplier(0.849) == pytest.approx(
            1.0 - (0.849 - 0.7) / 0.15
        )

    def test_at_kill_zone_boundary(self):
        assert similarity_multiplier(0.85) == 0.0

    def test_above_kill_zone(self):
        assert similarity_multiplier(0.9) == 0.0
        assert similarity_multiplier(1.0) == 0.0


class TestScoreCandidate:
    def test_fresh_no_fatigue(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        candidate = _make_candidate(score=80, created_at=now - timedelta(hours=5))
        result = score_candidate(candidate, now, [])
        assert result.final == 80.0  # 80 * 1.0 * 1.0 * 1.0
        assert result.raw == 80
        assert result.freshness == 1.0
        assert result.fatigue == 1.0
        assert result.similarity == 1.0

    def test_applies_freshness_decay(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        candidate = _make_candidate(score=80, created_at=now - timedelta(hours=30))
        result = score_candidate(candidate, now, [])
        assert result.final == pytest.approx(76.0)  # 80 * 0.95 * 1.0
        assert result.freshness == 0.95

    def test_applies_fatigue_penalty(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        candidate = _make_candidate(
            score=80,
            archetype=CandidateArchetype.META_TREND,
            created_at=now - timedelta(hours=5),
        )
        result = score_candidate(
            candidate, now, [CandidateArchetype.META_TREND]
        )
        assert result.final == pytest.approx(48.0)  # 80 * 1.0 * 0.6
        assert result.fatigue == 0.6

    def test_applies_similarity_penalty(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        candidate = _make_candidate(score=80, created_at=now - timedelta(hours=5))
        # 0.775 similarity → 0.5 multiplier
        scores = {candidate.id: 0.775}
        result = score_candidate(candidate, now, [], scores)
        assert result.final == pytest.approx(40.0)  # 80 * 1.0 * 1.0 * 0.5
        assert result.similarity == pytest.approx(0.5)

    def test_similarity_hard_veto(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        candidate = _make_candidate(score=100, created_at=now)
        scores = {candidate.id: 0.9}
        result = score_candidate(candidate, now, [], scores)
        assert result.final == 0.0
        assert result.similarity == 0.0

    def test_applies_both_freshness_and_fatigue(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        candidate = _make_candidate(
            score=100,
            archetype=CandidateArchetype.FRICTION_POINT,
            created_at=now - timedelta(hours=50),
        )
        result = score_candidate(
            candidate,
            now,
            [CandidateArchetype.FRICTION_POINT],
        )
        # 100 * 0.90 * 0.6 = 54.0
        assert result.final == pytest.approx(54.0)
        assert result.freshness == 0.90
        assert result.fatigue == 0.6


class TestSelectBestCandidate:
    def test_empty_list_returns_none(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        assert select_best_candidate([], now, []) is None

    def test_single_candidate(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        candidate = _make_candidate(score=50, created_at=now)
        winner, scored = select_best_candidate([candidate], now, [])
        assert winner.candidate is candidate
        assert len(scored) == 1

    def test_highest_raw_score_wins_when_no_modifiers(self):
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        low = _make_candidate(score=40, created_at=now)
        high = _make_candidate(score=90, created_at=now)
        winner, scored = select_best_candidate([low, high], now, [])
        assert winner.candidate is high
        assert scored[0].candidate is high
        assert scored[1].candidate is low

    def test_freshness_can_change_winner(self):
        """A fresher candidate with a lower raw score can beat a stale one."""
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        # Stale but high raw score: 90 * 0.90 = 81.0
        stale = _make_candidate(score=90, created_at=now - timedelta(hours=50))
        # Fresh but lower raw score: 85 * 1.0 = 85.0
        fresh = _make_candidate(score=85, created_at=now - timedelta(hours=5))
        winner, _ = select_best_candidate([stale, fresh], now, [])
        assert winner.candidate is fresh

    def test_fatigue_can_change_winner(self):
        """Archetype fatigue can make a lower-scored candidate win."""
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        # META_TREND with fatigue: 90 * 1.0 * 0.6 = 54.0
        fatigued = _make_candidate(
            score=90,
            archetype=CandidateArchetype.META_TREND,
            created_at=now,
        )
        # FRICTION_POINT, no fatigue: 60 * 1.0 * 1.0 = 60.0
        fresh_archetype = _make_candidate(
            score=60,
            archetype=CandidateArchetype.FRICTION_POINT,
            created_at=now,
        )
        winner, _ = select_best_candidate(
            [fatigued, fresh_archetype],
            now,
            [CandidateArchetype.META_TREND],
        )
        assert winner.candidate is fresh_archetype

    def test_combined_modifiers(self):
        """Test a realistic scenario with multiple candidates and modifiers."""
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        recently_published = [
            CandidateArchetype.META_TREND,
            CandidateArchetype.FRICTION_POINT,
        ]

        # META_TREND, fresh, heavy fatigue: 85 * 1.0 * 0.6 = 51.0
        a = _make_candidate(
            score=85,
            archetype=CandidateArchetype.META_TREND,
            created_at=now - timedelta(hours=5),
        )
        # FRICTION_POINT, slightly stale, moderate fatigue: 80 * 0.95 * 0.8 = 60.8
        b = _make_candidate(
            score=80,
            archetype=CandidateArchetype.FRICTION_POINT,
            created_at=now - timedelta(hours=25),
        )
        # RABBIT_HOLE, stale, no fatigue: 70 * 0.90 * 1.0 = 63.0
        c = _make_candidate(
            score=70,
            archetype=CandidateArchetype.RABBIT_HOLE,
            created_at=now - timedelta(hours=50),
        )

        winner, scored = select_best_candidate([a, b, c], now, recently_published)
        assert winner.candidate is c  # 63.0 beats 60.8 and 51.0
        assert [s.candidate for s in scored] == [c, b, a]  # sorted by final desc
