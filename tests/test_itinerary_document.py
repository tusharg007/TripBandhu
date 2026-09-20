from __future__ import annotations

import unittest
from io import BytesIO

from pypdf import PdfReader

from decimal import Decimal

from itinerary_document import ItineraryValidationError, build_itinerary_document, extract_budget_summary, require_exportable
from pdf_generator import build_professional_itinerary_pdf


COMPLETE_PLAN = """# Delhi to Jaipur Travel Proposal

### Day 1 - Arrival
- Morning: Travel from Delhi to Jaipur.
- Afternoon: Check in near Bani Park.
- Evening: Visit Hawa Mahal.

### Day 2 - Heritage and food
- Morning: Visit Amber Fort.
- Afternoon: Explore City Palace.
- Evening: Take a food walk.

### Day 3 - Return
- Morning: Visit Jantar Mantar.
- Afternoon: Return to Delhi.
- Evening: Keep departure flexible.
"""


class ItineraryDocumentTests(unittest.TestCase):
    def test_builds_complete_contiguous_document_from_day_sections(self):
        document = build_itinerary_document(
            thread_id="thread-1",
            markdown=COMPLETE_PLAN,
            constraints={"origin": "Delhi", "destination": "Jaipur", "duration": "3 days"},
            approved=True,
        )
        self.assertTrue(document.is_complete)
        self.assertTrue(document.approved)
        self.assertEqual([day.day for day in document.days], [1, 2, 3])
        self.assertEqual(document.days[0].morning, ["Travel from Delhi to Jaipur."])
        self.assertEqual(document.days[1].evening, ["Take a food walk."])
        self.assertTrue(document.content_hash)

    def test_missing_requested_day_is_not_exportable(self):
        document = build_itinerary_document(
            thread_id="thread-1",
            markdown=COMPLETE_PLAN.replace("### Day 3 - Return\n- Morning: Visit Jantar Mantar.\n- Afternoon: Return to Delhi.\n- Evening: Keep departure flexible.\n", ""),
            constraints={"origin": "Delhi", "destination": "Jaipur", "duration": "3 days"},
            approved=True,
        )
        self.assertFalse(document.approved)
        with self.assertRaises(ItineraryValidationError):
            require_exportable(document)

    def test_professional_pdf_preserves_all_days_and_proposal_warning(self):
        document = build_itinerary_document(
            thread_id="thread-1",
            markdown=COMPLETE_PLAN,
            constraints={"origin": "Delhi", "destination": "Jaipur", "duration": "3 days"},
            approved=True,
        )
        pdf = build_professional_itinerary_pdf(document)
        reader = PdfReader(BytesIO(pdf))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.assertGreaterEqual(len(reader.pages), 2)
        self.assertIn("day 1", text.casefold())
        self.assertIn("day 2", text.casefold())
        self.assertIn("day 3", text.casefold())
        self.assertIn("Not a booking confirmation", text)
        self.assertEqual(reader.metadata.title, "Delhi to Jaipur Travel Proposal")

    def test_budget_parser_uses_decimal_ranges_without_inventing_rows(self):
        budget = extract_budget_summary("""| Category | Estimate |
| --- | --- |
| Flights | ₹45,000 – ₹60,000 |
| Hotel | ₹18,500 |
| Narrative only | no number available |
""")
        self.assertEqual(len(budget.line_items), 2)
        self.assertEqual(budget.line_items[0].amount_low, Decimal("45000"))
        self.assertEqual(budget.line_items[0].amount_high, Decimal("60000"))
        self.assertEqual(budget.totals(), (Decimal("63500.00"), Decimal("78500.00")))
