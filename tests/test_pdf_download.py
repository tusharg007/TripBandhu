"""Regression coverage for the server-side itinerary PDF download."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
import unittest

from fastapi.testclient import TestClient
from pypdf import PdfReader

import app
from pdf_generator import PdfContentError, build_itinerary_pdf, normalize_pdf_text


SAMPLE_ITINERARY = """# Delhi to Jaipur Cultural Escape

## Trip Summary
**Travelers:** 2 adults
**Budget:** ₹25,000

## Day-by-Day Itinerary
### Day 1 - Arrival and the Pink City
- Take the morning train from New Delhi to Jaipur.
- Check in near Bani Park and visit Hawa Mahal.

### Day 2 - Forts and Local Food
1. Visit Amber Fort before the midday heat.
2. Try a guided food walk in the old city.

| Time | Plan | Estimated cost |
| --- | --- | --- |
| 08:00 | Amber Fort | ₹500 |
| 18:00 | Food walk | ₹1,500 |

> Provider prices can change; confirm before booking.

### Day 3 - Palaces and Craft Traditions
- Tour City Palace and Jantar Mantar with a licensed local guide.
- Visit a block-printing workshop and keep the evening free for Johari Bazaar.

### Day 4 - Pushkar Day Trip
- Leave Jaipur after breakfast for Pushkar by pre-booked cab.
- Visit the lake ghats respectfully and return before late evening.

### Day 5 - Hill Fort Views
- Start early at Jaigarh Fort and continue to Nahargarh Fort.
- Reserve sunset time at an authorized viewpoint and avoid isolated trails.

### Day 6 - Food and Leisure
- Join a morning cooking class focused on Rajasthani dishes.
- Keep the afternoon flexible for shopping, rest, or an optional museum visit.

### Day 7 - Return to Delhi
- Check out after breakfast and take the confirmed train to New Delhi.
- Keep a 60-minute buffer before departure and verify the platform in the rail app.

## Budget Guide
| Category | Planning allowance | Notes |
| --- | --- | --- |
| Intercity transport | ₹6,000 | Train and Pushkar day trip |
| Accommodation | ₹8,000 | Six nights, double occupancy |
| Food | ₹5,000 | Local restaurants and one cooking class |
| Activities | ₹3,500 | Forts, museums, and guide fees |
| Contingency | ₹2,500 | Approximately ten percent |

## Booking and Safety Notes
- Verify live train availability and provider prices before payment.
- Carry government-issued identification and digital copies of confirmations.
- Use registered transport after dark and keep emergency contacts accessible.
- Weather, opening hours, and local restrictions may change after this plan is generated.
"""


class PdfGeneratorTests(unittest.TestCase):
    def test_builds_readable_pdf_with_itinerary_content(self):
        pdf_bytes = build_itinerary_pdf(
            SAMPLE_ITINERARY,
            "Delhi to Jaipur Cultural Escape",
            generated_at=datetime(2026, 8, 24, 10, 30),
        )

        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertGreater(len(pdf_bytes), 2_000)

        reader = PdfReader(BytesIO(pdf_bytes))
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.assertGreaterEqual(len(reader.pages), 2)
        self.assertIn("Delhi to Jaipur Cultural Escape", extracted)
        self.assertIn("Amber Fort", extracted)
        self.assertIn("INR 25,000", extracted)
        self.assertIn("Page 1", extracted)
        self.assertIn("Page 2", extracted)
        self.assertEqual(reader.metadata.title, "Delhi to Jaipur Cultural Escape")

    def test_normalizes_currency_and_non_ascii_dashes(self):
        self.assertEqual(normalize_pdf_text("₹2.5 lakh — Delhi"), "INR 2.5 lakh - Delhi")

    def test_rejects_empty_final_plan(self):
        with self.assertRaises(PdfContentError):
            build_itinerary_pdf("   ")

    def test_concurrent_downloads_generate_independent_documents(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            documents = list(executor.map(
                lambda index: build_itinerary_pdf(
                    SAMPLE_ITINERARY,
                    f"TripBandhu Plan {index}",
                ),
                range(4),
            ))

        self.assertEqual(len(documents), 4)
        self.assertTrue(all(document.startswith(b"%PDF-") for document in documents))
        self.assertTrue(all(len(document) > 2_000 for document in documents))

    def test_api_returns_downloadable_pdf_with_security_headers(self):
        client = TestClient(app.app)
        response = client.post("/api/travel/download-pdf", json={
            "text": SAMPLE_ITINERARY,
            "title": "Delhi to Jaipur Cultural Escape",
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertTrue(response.content.startswith(b"%PDF-"))
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")

    def test_api_rejects_whitespace_only_content(self):
        client = TestClient(app.app)
        response = client.post("/api/travel/download-pdf", json={"text": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error_code"], "INVALID_PDF_CONTENT")


if __name__ == "__main__":
    unittest.main()
