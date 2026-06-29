"""Tests for Drive folder-id normalization (pure, no network)."""

from __future__ import annotations

from ingestion.drive_client import extract_folder_id

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
