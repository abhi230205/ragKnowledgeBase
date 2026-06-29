"""Incremental-sync diff: reconcile a live Drive listing against tracked state.

The answer to the brief's "if a document is deleted or edited, how is it handled".
PURE function — no Drive, Chroma, or DB imports — so it unit-tests in isolation.

Detection (Drive ids are stable; md5_checksum changes when bytes change):
    added     : id in listing, not tracked
    modified  : id tracked, content changed                  -> delete chunks + re-embed
    renamed   : id tracked, content same, file_name differs  -> metadata update only
    deleted   : tracked id absent from listing               -> delete chunks
    unchanged : id + content + name all match                -> skip

"Content changed" is md5_checksum inequality; when md5 is unavailable on either
side (some Drive items expose no checksum), it falls back to modifiedTime.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _is_modified(rec: dict, f) -> bool:
    """True if the file's content changed. md5 is primary; modifiedTime is the
    fallback when md5 is missing on either side (None != None must not read as a
    change, and a real edit must still be caught)."""
    old_md5 = rec.get("md5_checksum")
    new_md5 = getattr(f, "md5_checksum", None)
    if old_md5 is not None and new_md5 is not None:
        return old_md5 != new_md5
    old_mt = rec.get("modified_time")
    new_mt = getattr(f, "modified_time", None)
    if old_mt is not None and new_mt is not None:
        return old_mt != new_mt
    return False  # can't tell -> treat as unchanged (rename/skip handles the rest)


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
        elif _is_modified(rec, f):
            diff.modified.append(f)
        elif rec.get("file_name") != f.name:
            diff.renamed.append(f)
        else:
            diff.unchanged.append(f)

    for file_id in tracked:
        if file_id not in seen:
            diff.deleted.append(file_id)

    return diff
