"""Metadata-only regression checks for the current campaign boundary."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from phase1_codegen import campaign_tasks
from phase1_codegen.experience import EvolutionMemory


class CurrentCampaignContractTests(unittest.TestCase):
    def test_task_planner_has_only_same_environment_dispatch(self):
        audit = dict(selected_counts={}, checks={})
        for environment in ('alfworld', 'textworld'):
            with self.subTest(environment=environment), contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(campaign_tasks, 'build_alfworld_cross_type', return_value=audit) as alfworld, \
                 patch.object(campaign_tasks, 'build_textworld', return_value=audit) as textworld:
                campaign_tasks.main(['--out', '/unused-metadata-fixture', '--environment', environment])
            selected, other = (alfworld, textworld) if environment == 'alfworld' else (textworld, alfworld)
            self.assertEqual(selected.call_count, 1)
            other.assert_not_called()
        self.assertFalse(hasattr(campaign_tasks, 'build_alfworld'))

    def test_retired_cross_environment_planner_option_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), \
             patch.object(campaign_tasks, 'build_alfworld_cross_type') as build, \
             self.assertRaises(SystemExit) as error:
            campaign_tasks.main(['--out', '/unused-metadata-fixture', '--protocol', 'legacy_cross_environment'])
        self.assertEqual(error.exception.code, 2)
        build.assert_not_called()

    def test_old_configuration_memories_cannot_enter_current_design_library(self):
        for kind in (None, 'configuration_trial', 'legacy_configuration_trial'):
            record = dict(id='old-fixture', before={'retrieval': 'bm25'}, patch={}, after={}, evidence={})
            if kind is not None:
                record['kind'] = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'code_change_trial'):
                EvolutionMemory([record])

    def test_current_code_change_records_keep_positive_and_negative_evidence(self):
        records = [dict(id='current-' + str(accepted), kind='code_change_trial',
                        observed_context={'environment': 'alfworld', 'family': 'fixture'},
                        accepted=accepted, synthetic=True, code_diff='fixture code change')
                   for accepted in (True, False)]
        self.assertEqual(EvolutionMemory(records).records, records)
        with self.assertRaisesRegex(ValueError, 'Unknown fields'):
            EvolutionMemory([dict(records[0], before={'retrieval': 'bm25'})])


if __name__ == '__main__':
    unittest.main()
