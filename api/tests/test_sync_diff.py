"""Sync-diff tests — the pure reconciliation function (no Drive/Chroma/DB)."""

from __future__ import annotations

from types import SimpleNamespace

from ingestion.sync_diff import compute_diff


def _f(id, name, md5):
    return SimpleNamespace(id=id, name=name, md5_checksum=md5)


def test_compute_diff_classifies_all_cases():
    drive = [
        _f("new", "new.pdf", "h_new"),  # added
        _f("mod", "mod.pdf", "h_v2"),  # modified (md5 changed)
        _f("ren", "renamed.pdf", "h_same"),  # renamed (md5 same, name changed)
        _f("keep", "keep.pdf", "h_keep"),  # unchanged
    ]
    tracked = {
        "mod": {"md5_checksum": "h_v1", "file_name": "mod.pdf"},
        "ren": {"md5_checksum": "h_same", "file_name": "old.pdf"},
        "keep": {"md5_checksum": "h_keep", "file_name": "keep.pdf"},
        "gone": {"md5_checksum": "h_gone", "file_name": "gone.pdf"},  # deleted
    }

    diff = compute_diff(drive, tracked)

    assert [f.id for f in diff.added] == ["new"]
    assert [f.id for f in diff.modified] == ["mod"]
    assert [f.id for f in diff.renamed] == ["ren"]
    assert [f.id for f in diff.unchanged] == ["keep"]
    assert diff.deleted == ["gone"]


def test_compute_diff_empty_tracked_all_added():
    drive = [_f("a", "a.pdf", "1"), _f("b", "b.pdf", "2")]
    diff = compute_diff(drive, {})
    assert {f.id for f in diff.added} == {"a", "b"}
    assert diff.modified == diff.deleted == diff.renamed == diff.unchanged == []


def test_compute_diff_empty_drive_all_deleted():
    tracked = {"x": {"md5_checksum": "1", "file_name": "x.pdf"}}
    diff = compute_diff([], tracked)
    assert diff.deleted == ["x"]
    assert diff.added == diff.modified == diff.renamed == diff.unchanged == []


def _fm(id, name, md5, mt):
    return SimpleNamespace(id=id, name=name, md5_checksum=md5, modified_time=mt)


def test_modifiedtime_fallback_when_md5_absent():
    # md5 missing on both sides -> fall back to modifiedTime.
    tracked = {
        "edited": {"md5_checksum": None, "file_name": "e.pdf", "modified_time": "t1"},
        "same": {"md5_checksum": None, "file_name": "s.pdf", "modified_time": "t1"},
    }
    drive = [
        _fm("edited", "e.pdf", None, "t2"),  # modifiedTime changed -> modified
        _fm("same", "s.pdf", None, "t1"),  # modifiedTime same -> unchanged
    ]
    diff = compute_diff(drive, tracked)
    assert [f.id for f in diff.modified] == ["edited"]
    assert [f.id for f in diff.unchanged] == ["same"]
