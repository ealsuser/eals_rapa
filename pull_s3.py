#!/usr/bin/env python3
"""
Pull EALS-RAPA data from S3 buckets.

==== USAGE:
# Pull all sources
python pull_s3.py

# Pull specific sources
python pull_s3.py ALSFRS-R ROADS
python pull_s3.py ZephyrX Aura
"""

import argparse
import os
from datetime import date

import boto3

from config import Aural, Paths, Zephyrx

SOURCES = ['ALSFRS-R', 'ROADS', 'Demographics', 'ZephyrX', 'Aural']

LOG_PATH = os.path.join(os.path.dirname(__file__), 's3_pull.log')


def _download_file(s3, bucket, key, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    s3.download_file(bucket, key, dest_path)
    return 1


def _sync_prefix(s3, bucket, prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            rel_path = key[len(prefix):]
            if not rel_path:
                continue
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)
            count += 1
    return count


def pull_alsfrsr(s3):
    paths = Paths()
    return _download_file(s3, paths.s3_bucket, 'RAPA EAP ALSFRS Data.csv', paths.alsfrsr)


def pull_roads(s3):
    paths = Paths()
    return _download_file(s3, paths.s3_bucket, 'RAPA EAP ROADS Data.csv', paths.roads)


def pull_demographics(s3):
    paths = Paths()
    return _download_file(s3, paths.s3_bucket, 'RAPA EAP Demographics Data.csv', paths.demographics)


def pull_zephyrx(s3):
    zephyrx = Zephyrx()
    return _sync_prefix(s3, zephyrx.s3_bucket, 'tests/', zephyrx.raw)


def pull_aural(s3):
    aural = Aural()
    return _sync_prefix(s3, aural.s3_bucket, '', aural.raw)


PULL_FUNCS = {
    'ALSFRS-R': pull_alsfrsr,
    'ROADS': pull_roads,
    'Demographics': pull_demographics,
    'ZephyrX': pull_zephyrx,
    'Aural': pull_aural,
}


def _update_log(counts):
    lines = [f'# {date.today().isoformat()}\n']
    for source, count in counts.items():
        lines.append(f'  {source}: {count} file(s)\n')
    lines.append('\n')

    existing = ''
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            existing = f.read()

    with open(LOG_PATH, 'w') as f:
        f.writelines(lines)
        f.write(existing)


def main():
    parser = argparse.ArgumentParser(description='Pull EALS-RAPA data from S3.')
    parser.add_argument(
        'sources',
        nargs='*',
        metavar='SOURCE',
        help=f'Sources to pull: {", ".join(SOURCES)}. Defaults to all.',
    )
    args = parser.parse_args()
    if not args.sources:
        args.sources = SOURCES
    else:
        invalid = [s for s in args.sources if s not in SOURCES]
        if invalid:
            parser.error(f'invalid choice(s): {invalid}. Choose from: {", ".join(SOURCES)}')

    s3 = boto3.client('s3')
    counts = {}

    for source in args.sources:
        print(f'Pulling {source}...', end=' ', flush=True)
        count = PULL_FUNCS[source](s3)
        counts[source] = count
        print(f'{count} file(s)')

    _update_log(counts)
    print(f'\nLog updated: {LOG_PATH}')



if __name__ == '__main__':
    main()
