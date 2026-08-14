"""
test_evaluation_harness.py — Automated verification of the TripBandhu Evaluation Benchmark Harness.
"""

import asyncio
import sys
import unittest

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from eval.eval_dataset import BENCHMARK_DATASET
from eval.evaluator import TripBandhuEvaluator, EvalMetrics
from eval.trace_observability import validate_public_trace_entry, scan_trace_for_sensitive_data


class EvaluationHarnessTest(unittest.TestCase):
    """Verify evaluation harness execution and benchmark metrics."""

    def test_benchmark_dataset_integrity(self):
        """Dataset has all required categories and minimum case count."""
        self.assertGreaterEqual(len(BENCHMARK_DATASET), 15)
        categories = {case.category for case in BENCHMARK_DATASET}
        expected_categories = {"routing", "guardrail", "capability_permissions", "groundedness", "negative_control"}
        self.assertTrue(expected_categories.issubset(categories))

    def test_evaluator_fast_mode_metrics(self):
        """Fast evaluator runs across dataset and achieves 100% on all critical invariants."""
        evaluator = TripBandhuEvaluator()
        metrics: EvalMetrics = asyncio.run(evaluator.run_evaluation(is_fast_mode=True))

        self.assertEqual(metrics.failed_cases, 0, "Zero failures permitted in benchmark suite")
        self.assertEqual(metrics.routing_accuracy, 100.0)
        self.assertEqual(metrics.guardrail_precision, 100.0)
        self.assertEqual(metrics.capability_permission_compliance, 100.0)
        self.assertEqual(metrics.evidence_label_accuracy, 100.0)
        self.assertEqual(metrics.groundedness_anti_fabrication_rate, 100.0)
        self.assertEqual(metrics.hitl_protocol_adherence, 100.0)
        self.assertEqual(metrics.provider_degradation_safety, 100.0)
        self.assertEqual(metrics.thread_isolation_integrity, 100.0)
        self.assertEqual(metrics.cancellation_safety, 100.0)
        self.assertEqual(metrics.schema_drift_detection_rate, 100.0)
        self.assertEqual(metrics.llm_accounting_accuracy, 100.0)
        self.assertEqual(metrics.critical_invariants_pass_rate, 100.0)
        self.assertEqual(metrics.negative_controls_pass_rate, 100.0)
        self.assertEqual(metrics.trace_safety_compliance, 100.0)

    def test_trace_observability_schema_and_secret_scanning(self):
        """Validate trace entry schema and secret scanning detection."""
        clean_entry = {
            "specialist": "flight_agent",
            "capability": "FLIGHT_ROUTE_SEARCH",
            "server": "aviationstack",
            "status": "AVAILABLE",
            "latency_ms": 120,
            "source_count": 1,
            "error_code": None,
        }
        is_valid, msg = validate_public_trace_entry(clean_entry)
        self.assertTrue(is_valid, msg)

        leaky_trace = [
            clean_entry,
            {
                "specialist": "hotel_agent",
                "capability": "HOTEL_WEB_RESEARCH",
                "server": "tavily",
                "status": "AVAILABLE",
                "latency_ms": 95,
                "source_count": 1,
                "error_code": "tvly-secret12345678901234567890",
            },
        ]
        violations = scan_trace_for_sensitive_data(leaky_trace)
        self.assertEqual(len(violations), 1)
        self.assertIn("tvly-", violations[0])


if __name__ == "__main__":
    unittest.main()
