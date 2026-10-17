"""Persist cgroup memory evidence for externally killed training processes."""
import json
import os
from pathlib import Path
import sys
import time

target = Path(sys.argv[1])
parent = int(sys.argv[2])
cgroup = Path('/sys/fs/cgroup')
with target.open('a', buffering=1) as log:
    while True:
        try:
            os.kill(parent, 0)
        except ProcessLookupError:
            break
        row = {'time': time.time()}
        for key in ('memory.current', 'memory.high', 'memory.max', 'memory.events', 'memory.stat'):
            p = cgroup / key
            if p.exists():
                row[key] = p.read_text().strip()
        log.write(json.dumps(row) + '\n')
        time.sleep(15)
