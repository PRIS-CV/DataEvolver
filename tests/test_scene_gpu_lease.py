"""Use real file locks with supplied GPU observations; no GPU access."""
import os

import pytest

from dataevolver.workflows.scene.gpu_lease import lease_gpu


def test_native_gpu_lease_preserves_reservations_and_releases_lock(tmp_path):
    (tmp_path / 'runtime').mkdir()
    devices = [dict(index=0, uuid='GPU-reserved', free_mib=40000, utilization=0),
               dict(index=1, uuid='GPU-free', free_mib=30000, utilization=0)]
    first, lock = lease_gpu(tmp_path, devices, {'GPU-reserved'})
    try:
        assert first['uuid'] == 'GPU-free' and not first['shared_peer']
        with pytest.raises(RuntimeError, match='No GPU lease'):
            lease_gpu(tmp_path, devices, {'GPU-reserved'})
    finally:
        os.close(lock)
    second, lock = lease_gpu(tmp_path, devices, {'GPU-reserved'})
    os.close(lock)
    assert second['uuid'] == 'GPU-free'
