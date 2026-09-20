from __future__ import annotations

import asyncio
import unittest

from artifact_store import ArtifactStore
from itinerary_document import build_itinerary_document


MARKDOWN = """# Delhi to Jaipur
### Day 1 - Arrival
- Morning: Depart Delhi.
- Afternoon: Arrive Jaipur.
- Evening: Rest.
"""


class ArtifactStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = ArtifactStore()
        await self.store.initialize()
        self.document = build_itinerary_document(
            thread_id="thread-1",
            markdown=MARKDOWN,
            constraints={"origin": "Delhi", "destination": "Jaipur", "duration": "1 day"},
            approved=True,
        )

    async def test_owner_can_reuse_cached_export_without_rerendering(self):
        record = await self.store.save_approved("owner-a", self.document)
        calls = 0

        def render(document):
            nonlocal calls
            calls += 1
            return b"%PDF-test"

        _, first = await self.store.get_or_create_pdf("owner-a", record.plan_id, 1, "renderer-a", "asset-a", render)
        _, second = await self.store.get_or_create_pdf("owner-a", record.plan_id, 1, "renderer-a", "asset-a", render)
        self.assertEqual(first, second)
        self.assertEqual(calls, 1)

    async def test_other_session_cannot_read_plan_or_artifact(self):
        record = await self.store.save_approved("owner-a", self.document)
        self.assertIsNone(await self.store.get_owned_plan("owner-b", record.plan_id, 1))
        with self.assertRaises(PermissionError):
            await self.store.get_or_create_pdf("owner-b", record.plan_id, 1, "renderer-a", "asset-a", lambda _: b"%PDF-test")

    async def test_thread_owner_binding_is_session_bound(self):
        await self.store.bind_thread("owner-a", "thread-1")
        self.assertTrue(await self.store.owns_thread("owner-a", "thread-1"))
        self.assertFalse(await self.store.owns_thread("owner-b", "thread-1"))

    async def test_same_approved_version_is_idempotent_for_the_owner(self):
        first = await self.store.save_approved("owner-a", self.document)
        duplicate = self.document.model_copy(update={"plan_id": "plan_duplicate"})
        second = await self.store.save_approved("owner-a", duplicate)
        self.assertEqual(first.plan_id, second.plan_id)
