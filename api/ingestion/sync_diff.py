"""Incremental-sync diff: reconcile a live Drive listing against tracked state.

The answer to the brief's "if a document is deleted or edited, how is it handled".
PURE function — no Drive, Chroma, or DB imports — so it unit-tests in isolation.

Detection (Drive ids are stable; md5_checksum changes when bytes change):
    added     : id in listing, not tracked
    modified  : id tracked, md5_checksum differs            -> delete chunks + re-embed
    renamed   : id tracked, md5 same, file_name differs      -> metadata update only
    deleted   : tracked id absent from listing               -> delete chunks
    unchanged : id + md5 + name all match                    -> skip
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SyncDiff:
    """Reconciliation result. added/modified/renamed/unchanged hold drive-file
    objects; deleted holds tracked file_ids."""

    added: list = field(default_factory=list)
    modified: list = field(default_factory=list)
    renamed: list = field(default_factory=list)
    deleted: list = field(default_factory=list)
    unchanged: list = field(default_factory=list)


def compute_diff(drive_files, tracked: dict[str, dict]) -> SyncDiff:
    """Compute the sync diff.

    `drive_files`: iterable of objects with `.id`, `.name`, `.md5_checksum`
    (duck-typed; ingestion.drive_client.DriveFile satisfies this).
    `tracked`: mapping file_id -> {"md5_checksum": str|None, "file_name": str}.
    """
    diff = SyncDiff()
    seen: set[str] = set()

    for f in drive_files:
        seen.add(f.id)
        rec = tracked.get(f.id)
        if rec is None:
            diff.added.append(f)
        elif rec.get("md5_checksum") != f.md5_checksum:
            diff.modified.append(f)
        elif rec.get("file_name") != f.name:
            diff.renamed.append(f)
        else:
            diff.unchanged.append(f)

    for file_id in tracked:
        if file_id not in seen:
            diff.deleted.append(file_id)

    return diff
