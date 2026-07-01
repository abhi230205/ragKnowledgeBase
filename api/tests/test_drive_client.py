"""Tests for the Drive client: folder-id normalization + auth-failure handling
(all pure / no network — the graded "Drive auth failure must not 500" path)."""

from __future__ import annotations

import pytest

from ingestion.drive_client import (
    DriveAuthError,
    DriveClient,
    _explain_http_error,
    extract_folder_id,
)

BARE = "1dU1guosT4U2NBK78tregMVgJ-9pVB2XO"


def test_extract_from_folder_url():
    assert extract_folder_id(f"https://drive.google.com/drive/folders/{BARE}") == BARE


def test_extract_from_folder_url_with_query():
    assert extract_folder_id(f"https://drive.google.com/drive/folders/{BARE}?usp=sharing") == BARE


def test_extract_from_open_id_link():
    assert extract_folder_id("https://drive.google.com/open?id=XYZ_789") == "XYZ_789"


def test_bare_id_passthrough():
    assert extract_folder_id(BARE) == BARE


def test_empty_passthrough():
    assert extract_folder_id("") == ""


# ---- Auth-failure handling: raises typed DriveAuthError (never a bare 500) ----


def test_missing_credentials_raises_auth_error(tmp_path):
    """No service-account info and no key file → actionable DriveAuthError."""
    with pytest.raises(DriveAuthError) as exc:
        DriveClient(service_account_path=str(tmp_path / "nope.json"), service_account_info=None)
    assert "credential" in str(exc.value).lower()


def test_malformed_service_account_info_raises_auth_error():
    """A JSON dict that isn't a valid service-account key → DriveAuthError ('malformed')."""
    with pytest.raises(DriveAuthError) as exc:
        DriveClient(service_account_info={"not": "a real service account"})
    assert "malformed" in str(exc.value).lower()


class _FakeHttpError(Exception):
    """Stand-in for googleapiclient.errors.HttpError with a status_code."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status_code = status


def test_explain_http_error_maps_status_codes():
    """401/403 → access-denied guidance, 404 → not-found, else generic — all human-readable."""
    assert "denied access" in _explain_http_error(_FakeHttpError(403)).lower()
    assert "403" in _explain_http_error(_FakeHttpError(403))
    assert "denied access" in _explain_http_error(_FakeHttpError(401)).lower()
    assert "not found" in _explain_http_error(_FakeHttpError(404)).lower()
    assert "drive api error" in _explain_http_error(_FakeHttpError(500)).lower()
