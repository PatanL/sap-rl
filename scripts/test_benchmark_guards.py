"""Mocked safety gates: never execute systemctl, nvidia-smi or containers."""
import copy
import subprocess
import unittest

from benchmark_guards import LaunchBlocked, UNIT_FIELDS, verify_launch


def fixtures():
    request = dict(requestId='approved-request', invocationId='a'*32, containerId='b'*64,
                   targetUpdate=8650, checkpoint='/not-read/checkpoint.pt', guardServices=['training.service'])
    proof = {**request, 'kind': 'natural-training-chunk-v2', 'containerExitCode': 0,
             'containerRunning': False, 'containerFinishedAt': '2026-09-06T15:00:00Z',
             'checkpointAbsentAtRequest': True, 'launcherHeldAtCompletion': True,
             'checkpointUpdate': 8650, 'checkpointSHA256': 'c'*64,
             'finishedRecord': 'finished updates=8650 transitions=73000000 parameters=14,346,925 device=cuda',
             'shutdownInitiator': 'approved-request'}
    unit = dict(LoadState='loaded', ActiveState='inactive', SubState='dead', Result='success',
                ExecMainStatus='15', InvocationID='', MainPID='0')
    return request, proof, unit


class GuardsTest(unittest.TestCase):
    def runner(self, unit, *, query_failure=False, gpu='', gpu_failure=False):
        def run(command, **kwargs):
            if command[0] == 'systemctl':
                return subprocess.CompletedProcess(command, int(query_failure),
                    stdout='\n'.join(f'{k}={v}' for k, v in unit.items()), stderr='')
            self.assertEqual(command[0], 'nvidia-smi')
            return subprocess.CompletedProcess(command, int(gpu_failure), stdout=gpu, stderr='')
        return run

    def test_active_even_without_compute_pid_and_all_transitional_states_block(self):
        request, proof, unit = fixtures()
        for active, sub in [('active', 'running'), ('activating', 'start'), ('activating', 'auto-restart'),
                            ('deactivating', 'stop'), ('reloading', 'reload')]:
            with self.subTest(active=active, sub=sub), self.assertRaises(LaunchBlocked):
                verify_launch(request, proof, run=self.runner({**unit, 'ActiveState': active, 'SubState': sub}), check_files=False)

    def test_systemctl_failure_unknown_and_incomplete_responses_block(self):
        request, proof, unit = fixtures()
        for altered, failure in [(unit, True), ({**unit, 'LoadState': 'not-found'}, False), ({}, False)]:
            with self.subTest(unit=altered), self.assertRaises(LaunchBlocked):
                verify_launch(request, proof, run=self.runner(altered, query_failure=failure), check_files=False)

    def test_manual_stop_before_or_after_deadline_is_not_natural_completion(self):
        request, proof, unit = fixtures()
        for time_case in ('before-deadline', 'after-deadline'):
            with self.subTest(time=time_case), self.assertRaises(LaunchBlocked):
                verify_launch(request, None, run=self.runner({**unit, 'ExecMainStatus': '0'}), check_files=False)
        for key, value in [('launcherHeldAtCompletion', False), ('shutdownInitiator', 'manual'),
                           ('finishedRecord', ''), ('invocationId', 'd'*32), ('checkpointAbsentAtRequest', False)]:
            with self.subTest(key=key), self.assertRaises(LaunchBlocked):
                verify_launch(request, {**proof, key: value}, run=self.runner(unit), check_files=False)

    def test_failed_campaign_blocks(self):
        request, proof, unit = fixtures()
        with self.assertRaises(LaunchBlocked):
            verify_launch(request, proof, run=self.runner({**unit, 'Result': 'exit-code'}), check_files=False)
        with self.assertRaises(LaunchBlocked):
            verify_launch(request, {**proof, 'containerExitCode': 137}, run=self.runner(unit), check_files=False)

    def test_affirmative_natural_chunk_completion_allows_launch(self):
        request, proof, unit = fixtures()
        verify_launch(request, proof, run=self.runner(unit), check_files=False)

    def test_unrelated_gpu_owner_and_ownership_query_failure_block(self):
        request, proof, unit = fixtures()
        for gpu, failure in [('12345\n', False), ('', True)]:
            with self.subTest(gpu=gpu, failure=failure), self.assertRaises(LaunchBlocked):
                verify_launch(request, proof, run=self.runner(unit, gpu=gpu, gpu_failure=failure), check_files=False)


if __name__ == '__main__':
    unittest.main()
