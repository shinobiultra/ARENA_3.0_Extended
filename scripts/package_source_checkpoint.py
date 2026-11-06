"""Derive flat strict review bundles while retaining exact source containers.

Unsupported compression and raw payloads stay in the content-addressed original;
invalid text/images/JSON still fail rather than silently becoming external data.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
from pathlib import Path
import tarfile

from scripts import evidence_bundles as bundles


def package_checkpoint(path: Path, original: Path, repository_path: str) -> dict:
    """Package an independently preserved original; never rewrite its receipts."""
    if bundles.contract_path(path).exists():
        raise ValueError("existing contract must be reviewed, never overwritten")
    original_record = bundles.file_record(original)
    if bundles.file_record(path) != original_record:
        raise ValueError("source and preserved original differ")
    members, external, retained = {}, {}, []
    with tarfile.open(original, 'r:gz') as source:
        for member in source:
            bundles._safe_name(member.name)
            if not member.isfile() or member.issparse() or member.name in members:
                raise ValueError("unsupported or duplicate original member")
            stream = source.extractfile(member)
            record = {'sha256': hashlib.file_digest(stream, 'sha256').hexdigest(),
                      'size_bytes': member.size}
            members[member.name] = record
            lower = member.name.lower()
            compressed = lower.endswith('.gz') and not lower.endswith('.json.gz')
            if Path(lower).suffix in bundles.RAW_SUFFIXES or compressed:
                external[member.name] = {**record, 'source_archive_member': member.name,
                    'source_archive_sha256': original_record['sha256'],
                    'reason': 'Unsupported raw or compressed container retained byte-exactly in original archive.'}
            else:
                bundles.inspect_member(member.name, source.extractfile(member).read())
                retained.append(member.name)
    provenance = {'kind': 'historical_archive_packaging',
        'measurement_claim': 'Packaging only; original container and every original member remain byte-exact. No scientific rerun or acceptance.',
        'source_archive': {'absolute_path': str(original.resolve()),
                           'repository_path': repository_path, **original_record},
        'source_archive_members': members}
    temporary = path.with_name(path.name + '.packaging-tmp.tar.gz')
    if temporary.exists() or bundles.contract_path(temporary).exists():
        raise ValueError('temporary packaging path already exists')
    with temporary.open('wb') as output:
        with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as target:
                with tarfile.open(original, 'r:gz') as source:
                    for member in source:
                        if member.name in retained:
                            target.addfile(copy.copy(member), source.extractfile(member))
    # Fully validate candidate before replacing the publication container.
    bundles.seal_review_bundle(temporary, provenance=provenance, external_payloads=external)
    if bundles.file_record(original) != original_record or bundles.file_record(path) != original_record:
        raise ValueError('source changed during packaging')
    temporary.replace(path)
    # The candidate contract is retained as preparation evidence by the caller if desired.
    bundles.contract_path(temporary).unlink()
    contract = bundles.seal_review_bundle(path, provenance=provenance, external_payloads=external)
    assert bundles.resolve_historical_archive(path, original_record['sha256']) == original.resolve()
    return contract
