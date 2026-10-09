"""Serialize memory-heavy study jobs on the single GPU across study managers."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time


@contextmanager
def gpu_lease(label,device='cuda'):
    if not str(device).startswith('cuda'):
        yield
        return
    directory=Path(__file__).resolve().parents[1]/'artifacts/resources'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'gpu_work.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state={'owner_pid':os.getpid(),'label':label,'acquired_at':time.time(),'held':True}
        path=directory/'gpu_owner.json'
        path.write_text(json.dumps(state,indent=2))
        try:yield
        finally:
            state.update(held=False,released_at=time.time())
            path.write_text(json.dumps(state,indent=2))
            fcntl.flock(lock,fcntl.LOCK_UN)
