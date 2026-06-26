"""Incremental-sync diff: reconcile a live Drive listing against tracked state.

This is the answer to the brief's "if a document is deleted or edited, how is it
handled". Kept a PURE function (no Drive, no Chroma, no DB) so it can be unit
tested in isolation (test plan #10).

Detection rules (Drive ids are stable; md5_checksum changes when bytes change):
    added     : id in Drive listing, not in tracked state
    modified  : id present, md5_checksum differs from stored
    deleted   : tracked id absent from current listing
    renamed   : same id + same checksum, different name (metadata-only update)
    unchanged : id + checksum both match  -> skip (no re-embedding)

TODO (Phase 2): implement compute_diff returning the sets below.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SyncDiff:
    """Reconciliation result. Each list holds DriveFile / file_id as appropriate."""

    added: list = field(default_factory=list)
    modified: list = field(default_factory=list)
    deleted: list = field(default_factory=list)
    renamed: list = field(default_factory=list)
    unchanged: list = field(default_factory=list)


def compute_diff(drive_files, tracked) -> SyncDiff:
    """Compute the sync diff.

    `drive_files`: list of ingestion.drive_client.DriveFile (current listing).
    `tracked`: mapping file_id -> stored record (md5_checksum, file_name, ...).
    TODO: implement (Phase 2).
    """
    raise NotImplementedError("sync_diff.compute_diff — Phase 2")
