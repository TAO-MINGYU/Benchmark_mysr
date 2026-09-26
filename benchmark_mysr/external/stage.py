"""Verify and unpack frozen runtime bundles to private node-local storage."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path


def stage(bundles, cache, name):
    bundles, cache = Path(bundles), Path(cache)
    info = json.loads((bundles/'manifest.json').read_text())[name]
    destination = cache/info['sha256'][:16]
    destination.mkdir(parents=True,exist_ok=True)
    os.chmod(destination,0o700)
    with (destination/'stage.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        marker=destination/'verified.json'
        if not marker.exists():
            archive=bundles/(name+'.tar')
            h=hashlib.sha256()
            with archive.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):
                    h.update(block)
            if h.hexdigest()!=info['sha256']:
                raise ValueError('Runtime archive checksum mismatch')
            subprocess.run(['tar','-xf',str(archive),'-C',str(destination)],check=True)
            marker.write_text(json.dumps(info)+'\n')
    return destination


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundles',required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--name',required=True)
    a=p.parse_args()
    print(stage(a.bundles,a.cache,a.name))


if __name__=='__main__':
    main()
