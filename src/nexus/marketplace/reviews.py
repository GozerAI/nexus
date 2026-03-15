"""Review management for marketplace listings.

Handles creation, retrieval, aggregation, and moderation of reviews
across both the provider and agent marketplaces.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import Review

logger = logging.getLogger(__name__)

# Constraints
MIN_REVIEW_BODY_LENGTH = 10
MAX_REVIEW_BODY_LENGTH = 5000
MAX_REVIEWS_PER_USER_PER_LISTING = 1


@dataclass
class RatingSnapshot:
    """Pre-computed rating statistics for a listing."""

    listing_id: str = ""
    average_rating: float = 0.0
    total_reviews: int = 0
    rating_distribution: Dict[int, int] = field(
        default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "average_rating": round(self.average_rating, 2),
            "total_reviews": self.total_reviews,
            "rating_distribution": dict(self.rating_distribution),
        }


class ReviewManager:
    """Manages reviews and ratings for marketplace listings."""

    def __init__(self) -> None:
        # listing_id -> list[Review]
        self._reviews: Dict[str, List[Review]] = {}
        # listing_id -> RatingSnapshot (cached)
        self._snapshots: Dict[str, RatingSnapshot] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def submit_review(
        self,
        listing_id: str,
        reviewer: str,
        rating: int,
        title: str = "",
        body: str = "",
        verified: bool = False,
    ) -> Review:
        """Submit a new review for a listing.

        Raises:
            ValueError: on invalid input or duplicate review.
        """
        if not listing_id:
            raise ValueError("listing_id is required")
        if not reviewer:
            raise ValueError("reviewer is required")
        if not 1 <= rating <= 5:
            raise ValueError(f"rating must be 1-5, got {rating}")
        if body and len(body) < MIN_REVIEW_BODY_LENGTH:
            raise ValueError(
                f"Review body must be at least {MIN_REVIEW_BODY_LENGTH} characters"
            )
        if len(body) > MAX_REVIEW_BODY_LENGTH:
            raise ValueError(
                f"Review body must be at most {MAX_REVIEW_BODY_LENGTH} characters"
            )

        existing = self._reviews.get(listing_id, [])
        user_reviews = [r for r in existing if r.reviewer == reviewer]
        if len(user_reviews) >= MAX_REVIEWS_PER_USER_PER_LISTING:
            raise ValueError(
                f"User {reviewer!r} already reviewed listing {listing_id!r}"
            )

        review = Review(
            listing_id=listing_id,
            reviewer=reviewer,
            rating=rating,
            title=title,
            body=body,
            verified=verified,
        )

        self._reviews.setdefault(listing_id, []).append(review)
        self._invalidate_snapshot(listing_id)
        logger.debug("Review %s submitted for listing %s", review.id, listing_id)
        return review

    def update_review(
        self,
        review_id: str,
        rating: Optional[int] = None,
        title: Optional[str] = None,
        body: Optional[str] = None,
    ) -> Review:
        """Update an existing review.

        Raises:
            KeyError: if review not found.
            ValueError: on invalid input.
        """
        review = self._find_review(review_id)
        if rating is not None:
            if not 1 <= rating <= 5:
                raise ValueError(f"rating must be 1-5, got {rating}")
            review.rating = rating
        if title is not None:
            review.title = title
        if body is not None:
            if body and len(body) < MIN_REVIEW_BODY_LENGTH:
                raise ValueError(
                    f"Review body must be at least {MIN_REVIEW_BODY_LENGTH} characters"
                )
            review.body = body
        review.updated_at = time.time()
        self._invalidate_snapshot(review.listing_id)
        return review

    def delete_review(self, review_id: str) -> None:
        """Delete a review by ID.

        Raises:
            KeyError: if review not found.
        """
        review = self._find_review(review_id)
        self._reviews[review.listing_id].remove(review)
        self._invalidate_snapshot(review.listing_id)
        logger.debug("Review %s deleted", review_id)

    def mark_helpful(self, review_id: str) -> Review:
        """Increment the helpful counter on a review.

        Raises:
            KeyError: if review not found.
        """
        review = self._find_review(review_id)
        review.helpful_count += 1
        return review

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_reviews(
        self,
        listing_id: str,
        *,
        sort_by: str = "newest",
        min_rating: int = 1,
        verified_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Review]:
        """Retrieve reviews for a listing with filtering and sorting."""
        reviews = list(self._reviews.get(listing_id, []))

        # Filter
        if min_rating > 1:
            reviews = [r for r in reviews if r.rating >= min_rating]
        if verified_only:
            reviews = [r for r in reviews if r.verified]

        # Sort
        if sort_by == "newest":
            reviews.sort(key=lambda r: r.created_at, reverse=True)
        elif sort_by == "oldest":
            reviews.sort(key=lambda r: r.created_at)
        elif sort_by == "highest":
            reviews.sort(key=lambda r: r.rating, reverse=True)
        elif sort_by == "lowest":
            reviews.sort(key=lambda r: r.rating)
        elif sort_by == "helpful":
            reviews.sort(key=lambda r: r.helpful_count, reverse=True)

        return reviews[offset : offset + limit]

    def get_rating_snapshot(self, listing_id: str) -> RatingSnapshot:
        """Get cached rating snapshot for a listing."""
        if listing_id not in self._snapshots:
            self._rebuild_snapshot(listing_id)
        return self._snapshots[listing_id]

    def get_review_by_id(self, review_id: str) -> Review:
        """Retrieve a single review by its ID.

        Raises:
            KeyError: if not found.
        """
        return self._find_review(review_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_review(self, review_id: str) -> Review:
        for reviews in self._reviews.values():
            for r in reviews:
                if r.id == review_id:
                    return r
        raise KeyError(f"Review {review_id!r} not found")

    def _invalidate_snapshot(self, listing_id: str) -> None:
        self._snapshots.pop(listing_id, None)

    def _rebuild_snapshot(self, listing_id: str) -> None:
        reviews = self._reviews.get(listing_id, [])
        dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in reviews:
            dist[r.rating] += 1
        total = len(reviews)
        avg = sum(r.rating for r in reviews) / total if total else 0.0
        self._snapshots[listing_id] = RatingSnapshot(
            listing_id=listing_id,
            average_rating=avg,
            total_reviews=total,
            rating_distribution=dist,
        )
