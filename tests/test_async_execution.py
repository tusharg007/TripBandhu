
"""TF1: Async execution compliance and lifecycle tests."""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class AsyncExecutionTest(unittest.TestCase):

    def _source(self, filename: str) -> str:
        return ROOT.joinpath(filename).read_text(encoding='utf-8')

    def test_no_nest_asyncio_in_app(self):
        """app.py must not import or apply nest_asyncio."""
        src = self._source('app.py')
        self.assertNotIn('nest_asyncio', src,
                         "app.py must not use nest_asyncio (TF1 uses native async)")

    def test_no_nest_asyncio_in_backend(self):
        """backend.py must not import or apply nest_asyncio."""
        src = self._source('backend.py')
        self.assertNotIn('nest_asyncio', src,
                         "backend.py must not use nest_asyncio")

    def test_no_asyncio_run_in_specialist_nodes(self):
        """Specialist nodes must NOT call asyncio.run() — they must be awaited."""
        src = self._source('backend.py')
        tree = ast.parse(src)
        specialist_funcs = {
            'flight_agent', 'hotel_agent', 'weather_agent',
            'budget_agent', 'itinerary_agent', 'itinerary_revision_agent',
            'human_review_node', 'final_agent', 'supervisor_agent',
            'review_limit_reached_agent',
        }
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name in specialist_funcs:
                for child in ast.walk(node):
                    if (isinstance(child, ast.Call) and
                            isinstance(child.func, ast.Attribute) and
                            child.func.attr == 'run' and
                            isinstance(child.func.value, ast.Name) and
                            child.func.value.id == 'asyncio'):
                        violations.append(node.name)
        self.assertEqual(violations, [],
                         f"asyncio.run() found inside specialist nodes: {violations}")

    def test_specialist_nodes_are_async(self):
        """All specialist graph nodes must be defined as async functions."""
        src = self._source('backend.py')
        tree = ast.parse(src)
        async_funcs = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
        }
        required_async = [
            'supervisor_agent', 'guardrail_blocked_agent',
            'flight_agent', 'hotel_agent', 'weather_agent',
            'budget_agent', 'itinerary_agent', 'itinerary_revision_agent',
            'human_review_node', 'final_agent', 'review_limit_reached_agent',
        ]
        for fn in required_async:
            self.assertIn(fn, async_funcs, f"{fn} must be defined as async def")

    def test_run_travel_agent_is_async(self):
        """run_travel_agent and resume_travel_agent must be async."""
        src = self._source('backend.py')
        tree = ast.parse(src)
        async_funcs = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
        }
        self.assertIn('run_travel_agent', async_funcs)
        self.assertIn('resume_travel_agent', async_funcs)

    def test_app_has_lifespan_and_travel_service(self):
        """app.py must use lifespan context manager and bind travel_service to app.state."""
        src = self._source('app.py')
        self.assertIn('lifespan', src)
        self.assertIn('app.state.travel_service', src)
        self.assertNotIn('PostgresSaver.setup()', src)

    def test_separable_graph_construction(self):
        """backend.py must expose build_travel_graph and create_travel_service for separable testing."""
        import backend
        from langgraph.checkpoint.memory import InMemorySaver
        service = backend.create_travel_service(checkpointer=InMemorySaver())
        self.assertIsInstance(service, backend.TravelAgentService)


if __name__ == "__main__":
    unittest.main()
