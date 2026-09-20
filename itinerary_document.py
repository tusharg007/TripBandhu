"""Canonical, validated itinerary documents used for approval and export.

The browser may render Markdown, but approval, versioning, arithmetic, and PDF
generation operate on this typed document.  It intentionally treats provider
results as planning evidence rather than booking confirmations.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


DOCUMENT_SCHEMA_VERSION = "1.0"
_DAY_HEADING = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:day\s*)(\d{1,2})\s*(?:[-:|.]\s*)?(.*)$",
    re.IGNORECASE,
)
_DURATION = re.compile(r"\b(\d{1,2})\s*(?:day|days|night|nights)\b", re.IGNORECASE)
_MONEY_VALUE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)")


class ItineraryValidationError(ValueError):
    """Raised when a proposed itinerary cannot safely become an approved export."""


class DocumentWarning(BaseModel):
    code: str
    message: str
    severity: str = "warning"


class SourceNote(BaseModel):
    provider: str
    title: str
    url: str | None = None
    retrieved_at: str | None = None
    evidence_kind: str = "UNAVAILABLE"
    freshness: str = "UNKNOWN"


class BudgetLineItem(BaseModel):
    category: str
    amount_low: Decimal | None = None
    amount_high: Decimal | None = None
    currency: str = "INR"
    unit: str = "per group"
    estimate_status: str = "MODEL_ESTIMATE"
    notes: str = ""

    @model_validator(mode="after")
    def validate_range(self):
        if self.amount_low is not None and self.amount_high is not None and self.amount_low > self.amount_high:
            raise ValueError("Budget lower bound cannot exceed the upper bound.")
        return self


class BudgetSummary(BaseModel):
    primary_currency: str = "INR"
    line_items: list[BudgetLineItem] = Field(default_factory=list)
    contingency_rate: Decimal = Decimal("0.10")
    exchange_rate: Decimal | None = None
    exchange_rate_as_of: str | None = None
    exchange_rate_source: str | None = None

    def totals(self) -> tuple[Decimal, Decimal]:
        low = sum((item.amount_low or Decimal("0")) for item in self.line_items)
        high = sum((item.amount_high or item.amount_low or Decimal("0")) for item in self.line_items)
        return (
            low.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            high.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        )


class DailyPlan(BaseModel):
    day: int = Field(ge=1, le=60)
    title: str
    date: str | None = None
    morning: list[str] = Field(default_factory=list)
    afternoon: list[str] = Field(default_factory=list)
    evening: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    accommodation_note: str | None = None
    transport_note: str | None = None
    source_markdown: str = ""


class ItineraryDocument(BaseModel):
    schema_version: str = DOCUMENT_SCHEMA_VERSION
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid4().hex}")
    version: int = Field(default=1, ge=1)
    thread_id: str
    title: str
    origin: str | None = None
    destinations: list[str] = Field(default_factory=list)
    duration_days: int | None = Field(default=None, ge=1, le=60)
    travel_dates: str | None = None
    traveler_count: int | None = Field(default=None, ge=1, le=50)
    room_assumptions: str = "Not specified; availability and occupancy must be confirmed."
    days: list[DailyPlan] = Field(default_factory=list)
    budget: BudgetSummary = Field(default_factory=BudgetSummary)
    sources: list[SourceNote] = Field(default_factory=list)
    warnings: list[DocumentWarning] = Field(default_factory=list)
    approved: bool = False
    approved_at: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_markdown: str
    content_hash: str = ""

    @field_validator("title", "thread_id", "raw_markdown")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("A non-empty itinerary field is required.")
        return cleaned

    @model_validator(mode="after")
    def validate_days_and_hash(self):
        day_numbers = [day.day for day in self.days]
        if day_numbers and day_numbers != list(range(1, len(day_numbers) + 1)):
            raise ValueError("Itinerary day numbers must be contiguous and start at 1.")
        # A draft may legitimately be incomplete while it is being revised.  The
        # export boundary (`require_exportable`) rejects that state explicitly;
        # raising here would discard the useful structured validation warning.
        if not self.content_hash:
            self.content_hash = canonical_content_hash(self)
        return self

    @property
    def is_complete(self) -> bool:
        return bool(self.days) and (self.duration_days is None or len(self.days) == self.duration_days)


def _clean_markdown_line(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip(" #-\t"))


def _section_items(lines: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    buckets: dict[str, list[str]] = {"morning": [], "afternoon": [], "evening": [], "highlights": []}
    active = "highlights"
    for line in lines:
        cleaned = _clean_markdown_line(line)
        if not cleaned:
            continue
        lowered = cleaned.casefold()
        period_match = re.match(r"^(morning|afternoon|evening|night)\s*[:|-]\s*(.*)$", cleaned, re.IGNORECASE)
        if period_match:
            period = period_match.group(1).casefold()
            active = "evening" if period == "night" else period
            remainder = period_match.group(2).strip()
            if remainder:
                buckets[active].append(remainder[:500])
            continue
        if lowered in {"morning", "afternoon", "evening", "night"}:
            active = "evening" if lowered == "night" else lowered
            continue
        if cleaned.startswith("|") or set(cleaned) <= {"|", "-", ":", " "}:
            continue
        buckets[active].append(cleaned[:500])
    return buckets["morning"], buckets["afternoon"], buckets["evening"], buckets["highlights"]


def extract_duration_days(value: str | None) -> int | None:
    match = _DURATION.search(str(value or ""))
    if not match:
        return None
    return int(match.group(1))


def extract_daily_plans(markdown: str) -> list[DailyPlan]:
    """Extract explicit Day 1..N sections without inventing activities."""
    lines = str(markdown or "").replace("\r\n", "\n").split("\n")
    found: list[tuple[int, str, int]] = []
    for index, line in enumerate(lines):
        match = _DAY_HEADING.match(line)
        if match:
            found.append((int(match.group(1)), _clean_markdown_line(match.group(2)) or f"Day {match.group(1)}", index))

    days: list[DailyPlan] = []
    for position, (number, title, start) in enumerate(found):
        end = found[position + 1][2] if position + 1 < len(found) else len(lines)
        section_lines = lines[start + 1:end]
        morning, afternoon, evening, highlights = _section_items(section_lines)
        days.append(
            DailyPlan(
                day=number,
                title=title,
                morning=morning,
                afternoon=afternoon,
                evening=evening,
                highlights=highlights,
                source_markdown="\n".join(section_lines).strip(),
            )
        )
    return days


def _source_notes(evidence_store: dict[str, Any] | None) -> list[SourceNote]:
    notes: list[SourceNote] = []
    for evidence in (evidence_store or {}).values():
        if not isinstance(evidence, dict):
            continue
        for source in evidence.get("sources") or []:
            if not isinstance(source, dict):
                continue
            title = str(source.get("title") or "").strip()
            provider = str(source.get("provider") or "TripBandhu evidence").strip()
            if title:
                notes.append(
                    SourceNote(
                        provider=provider,
                        title=title,
                        url=source.get("url"),
                        retrieved_at=source.get("observed_at") or source.get("retrieved_at"),
                        evidence_kind=str(source.get("evidence_kind") or "UNAVAILABLE"),
                        freshness=str(source.get("freshness") or "UNKNOWN"),
                    )
                )
    return notes[:20]


def extract_budget_summary(markdown: str | None) -> BudgetSummary:
    """Conservatively parse explicit Markdown budget rows without inventing costs.

    Only rows containing a category and one or two numeric amounts are retained.
    They remain ``MODEL_ESTIMATE`` values unless a provider-backed source is added
    to the canonical document separately.
    """
    line_items: list[BudgetLineItem] = []
    seen_categories: set[str] = set()
    for raw_line in str(markdown or "").splitlines():
        if "|" not in raw_line:
            continue
        cells = [_clean_markdown_line(cell) for cell in raw_line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        category, amount_cell = cells[0], cells[1]
        normalized_category = category.casefold()
        if not category or normalized_category in {"category", "item", "description", "cost", "estimated cost"}:
            continue
        if set(category) <= {"-", ":", " "} or set(amount_cell) <= {"-", ":", " "}:
            continue
        amounts = [decimal_from_text(match.group(1)) for match in _MONEY_VALUE.finditer(amount_cell)]
        values = [amount for amount in amounts if amount is not None]
        if not values or any(amount < 0 or amount > Decimal("100000000") for amount in values):
            continue
        if normalized_category in seen_categories:
            continue
        seen_categories.add(normalized_category)
        currency = "INR" if "₹" in amount_cell or "inr" in amount_cell.casefold() else "USD" if "$" in amount_cell or "usd" in amount_cell.casefold() else "INR"
        line_items.append(
            BudgetLineItem(
                category=category[:120],
                amount_low=values[0],
                amount_high=values[1] if len(values) > 1 else values[0],
                currency=currency,
                notes="Parsed from the approved planning budget; verify before booking.",
            )
        )
    primary_currency = line_items[0].currency if line_items and len({item.currency for item in line_items}) == 1 else "INR"
    return BudgetSummary(primary_currency=primary_currency, line_items=line_items)


def canonical_content_hash(document: ItineraryDocument) -> str:
    payload = document.model_dump(exclude={"content_hash", "plan_id", "created_at", "approved_at"}, mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_itinerary_document(
    *,
    thread_id: str,
    markdown: str,
    constraints: dict[str, Any] | None = None,
    evidence_store: dict[str, Any] | None = None,
    budget_markdown: str | None = None,
    version: int = 1,
    approved: bool = False,
) -> ItineraryDocument:
    """Create a canonical planning document from reviewed itinerary Markdown.

    Missing facts remain assumptions or warnings; this function never fills gaps
    with model-generated values.
    """
    constraints = constraints or {}
    clean_markdown = str(markdown or "").strip()
    if not clean_markdown:
        raise ItineraryValidationError("A reviewed itinerary is required.")

    days = extract_daily_plans(clean_markdown)
    duration = extract_duration_days(constraints.get("duration"))
    destination = str(constraints.get("destination") or "").strip()
    warnings: list[DocumentWarning] = [
        DocumentWarning(
            code="NOT_A_BOOKING_CONFIRMATION",
            message="This travel proposal is not a booking confirmation. Verify availability, prices, and schedules before payment.",
        )
    ]
    if not duration:
        warnings.append(DocumentWarning(
            code="UNDATED_DURATION",
            message="Trip duration was not explicit; day labels are presented as a planning sequence.",
        ))
    if not days:
        warnings.append(DocumentWarning(
            code="MISSING_DAY_STRUCTURE",
            message="The approved text did not contain explicit Day 1..N headings; export is unavailable until the itinerary is revised.",
            severity="error",
        ))
    elif duration and len(days) != duration:
        warnings.append(DocumentWarning(
            code="DAY_COUNT_MISMATCH",
            message=f"The request requires {duration} days but the reviewed text contains {len(days)} explicit day sections.",
            severity="error",
        ))

    title = next(
        (
            _clean_markdown_line(line)
            for line in clean_markdown.splitlines()
            if line.lstrip().startswith("#") and "day" not in line.casefold()
        ),
        "TripBandhu Travel Proposal",
    )
    title = title or (f"{constraints.get('origin') or 'India'} to {destination}" if destination else "TripBandhu Travel Proposal")
    budget = extract_budget_summary(budget_markdown)
    if budget.line_items and any(item.currency != budget.primary_currency for item in budget.line_items):
        warnings.append(DocumentWarning(
            code="MIXED_BUDGET_CURRENCIES",
            message="Budget lines use multiple currencies and are not summed into one total.",
        ))
    document = ItineraryDocument(
        thread_id=thread_id,
        version=version,
        title=title,
        origin=str(constraints.get("origin") or "").strip() or None,
        destinations=[destination] if destination else [],
        duration_days=duration,
        travel_dates=str(constraints.get("dates") or "").strip() or None,
        traveler_count=_safe_positive_int(constraints.get("traveler_count")),
        days=days,
        budget=budget,
        sources=_source_notes(evidence_store),
        warnings=warnings,
        approved=approved,
        approved_at=datetime.now(timezone.utc).isoformat() if approved else None,
        raw_markdown=clean_markdown,
    )
    if any(warning.severity == "error" for warning in document.warnings):
        # Keep the document inspectable, but prevent it becoming an export artifact.
        document.approved = False
        document.content_hash = canonical_content_hash(document)
    return document


def require_exportable(document: ItineraryDocument) -> ItineraryDocument:
    errors = [warning.message for warning in document.warnings if warning.severity == "error"]
    if not document.approved:
        errors.append("The itinerary has not been approved as an exportable version.")
    if not document.is_complete:
        errors.append("The itinerary is missing complete, contiguous day sections.")
    if errors:
        raise ItineraryValidationError(" ".join(errors))
    return document


def _safe_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def decimal_from_text(value: Any) -> Decimal | None:
    """Parse a numeric amount deterministically for budget validation helpers."""
    cleaned = re.sub(r"[^0-9.-]", "", str(value or ""))
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None
