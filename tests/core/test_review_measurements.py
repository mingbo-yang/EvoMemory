"""CPU checks for evidence units, unknown values and exact grouping."""
import importlib.util
from pathlib import Path
import sys
import unittest

from phase1_codegen.roles import reviewer as review


def row(task='t', unit='one_actor_memory_slot', tokens=15, actual=True, match=None):
    return dict(task_id=task, unit=unit, read_tokens=tokens,
                actual_trace_available=actual, replay_matches_actual=match)


class MeasurementTests(unittest.TestCase):
    def test_aggregate_is_exact_and_never_invents_missing_measurements(self):
        result = review._aggregate_read_facts([row(tokens=15), row(tokens=None, actual=False)])[0]
        self.assertEqual(result['read_tokens'], 15)
        self.assertEqual(result['read_count'], 2)
        self.assertEqual(result['missing_token_measurements'], 1)
        self.assertEqual(result['actual_trace_rows'], 1)
        self.assertFalse(result['actual_trace_available'])
        self.assertIsNone(result['replay_matches_actual'])
        self.assertEqual(result['replay_comparison_rows'], 0)

    def test_no_numeric_measurements_remain_unknown(self):
        result = review._aggregate_read_facts([row(tokens=None), row(tokens=None)])[0]
        self.assertIsNone(result['read_tokens'])
        self.assertIsNone(result['minimum_row_read_tokens'])
        self.assertIsNone(result['maximum_row_read_tokens'])
        self.assertEqual(result['missing_token_measurements'], 2)

    def test_different_tasks_and_units_are_not_merged(self):
        rows = [row(task='a'), row(task='b'), row(task='a', unit='task_start_readout')]
        self.assertEqual(review._aggregate_read_facts(rows), rows)

    def test_unknown_scopes_are_not_merged(self):
        for rows in ([row(task=None), row(task=None)], [row(unit=None), row(unit=None)]):
            self.assertEqual(review._aggregate_read_facts(rows), rows)

    def test_partial_replay_comparison_has_explicit_denominator(self):
        result = review._aggregate_read_facts([row(match=True), row(match=False), row(match=None)])[0]
        self.assertEqual(result['replay_comparison_rows'], 2)
        self.assertEqual(result['replay_match_rows'], 1)
        self.assertFalse(result['replay_matches_actual'])


if __name__ == '__main__':
    unittest.main()
