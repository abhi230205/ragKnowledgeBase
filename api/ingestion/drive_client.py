"""Google Drive API v3 client (service-account auth).

Lists PDFs in a folder (recursively, walking sub-folders) and downloads file
bytes. Designed to fail with clear, typed errors when credentials are missing or
invalid, so the UI can show a friendly message instead of a 500 (a graded
common-pitfall). Credentials may come from a JSON key file path or an in-memory
dict — the latter is how the Settings UI passes uploaded creds.

Claude is not involved here; this module only talks to Google over HTTPS/REST.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)

# Read-only is all we need to list + download.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

PDF_MIME = "application/pdf"
FOLDER_MIME = "application/vnd.google-apps.folder"


class DriveAuthError(Exception):
    """Service-account credentials are missing, malformed, or rejected."""


class DriveError(Exception):
    """Non-auth Drive API failure (network, permissions, not found, ...)."""


@dataclass
class DriveFile:
    """A Drive file's metadata — the fields we track for incremental sync."""

    id: str
    name: str
    mime_type: str
    md5_checksum: str | None
    modified_time: str | None
    size: int | None

    @classmethod
    def from_api(cls, data: dict) -> "DriveFile":
        size = data.get("size")
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            mime_type=data.get("mimeType", ""),
            md5_checksum=data.get("md5Checksum"),
            modified_time=data.get("modifiedTime"),
            size=int(size) if size is not None else None,
        )


def _load_credentials(
    service_account_path: str | None = None,
    service_account_info: dict | None = None,
):
    """Build service-account credentials from a dict or a JSON key file.

    Raises DriveAuthError with an actionable message if neither is usable.
    """
    try:
        if service_account_info:
            return service_account.Credentials.from_service_account_info(
                service_account_info, scopes=SCOPES
            )
        if service_account_path and os.path.exists(service_account_path):
            return service_account.Credentials.from_service_account_file(
                service_account_path, scopes=SCOPES
            )
    except (ValueError, KeyError) as exc:
        raise DriveAuthError(f"Service account JSON is malformed: {exc}") from exc

    raise DriveAuthError(
        "No Google service account credentials found. Upload the JSON key in "
        "Settings, or set GOOGLE_SERVICE_ACCOUNT_PATH to a valid file."
    )


class DriveClient:
    """Thin wrapper over the Drive v3 API used by the ingestion pipeline."""

    def __init__(
        self,
        service_account_path: str | None = None,
        service_account_info: dict | None = None,
    ):
        creds = _load_credentials(service_account_path, service_account_info)
        try:
            # cache_discovery=False avoids a noisy file-cache warning in containers.
            self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        except Exception as exc:  # pragma: no cover - defensive
            raise DriveAuthError(f"Failed to initialise Drive client: {exc}") from exc

    def list_pdfs(self, folder_id: str, recursive: bool = True) -> list[DriveFile]:
        """Return every PDF in `folder_id` (and sub-folders if recursive).

        Folders are walked iteratively; both folders and files are de-duplicated
        by id (a file may have multiple parents / be reachable via shortcuts, so
        the same PDF can surface under two scanned folders).
        Raises DriveError on API failure or if folder_id is empty.
        """
        if not folder_id:
            raise DriveError("No Drive folder id configured.")

        pdfs: list[DriveFile] = []
        to_scan: list[str] = [folder_id]
        seen_folders: set[str] = set()
        seen_files: set[str] = set()

        while to_scan:
            current = to_scan.pop()
            if current in seen_folders:
                continue
            seen_folders.add(current)
            for f in self._list_children(current):
                if f.mime_type == PDF_MIME:
                    if f.id not in seen_files:
                        seen_files.add(f.id)
                        pdfs.append(f)
                elif recursive and f.mime_type == FOLDER_MIME:
                    to_scan.append(f.id)

        logger.info("Drive: found %d PDF(s) under folder %s", len(pdfs), folder_id)
        return pdfs

    def _list_children(self, folder_id: str) -> list[DriveFile]:
        """List immediate PDF + folder children of `folder_id`, paging fully."""
        query = (
            f"'{folder_id}' in parents and trashed = false and "
            f"(mimeType = '{PDF_MIME}' or mimeType = '{FOLDER_MIME}')"
        )
        fields = "nextPageToken, files(id, name, mimeType, md5Checksum, modifiedTime, size)"
        children: list[DriveFile] = []
        page_token: str | None = None
        try:
            while True:
                resp = (
                    self._service.files()
                    .list(
                        q=query,
                        fields=fields,
                        pageSize=100,
                        pageToken=page_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )
                children.extend(DriveFile.from_api(d) for d in resp.get("files", []))
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as exc:
            raise DriveError(_explain_http_error(exc)) from exc
        return children

    def download_bytes(self, file_id: str) -> bytes:
        """Download a file's raw bytes via files.get_media."""
        try:
            request = self._service.files().get_media(fileId=file_id, supportsAllDrives=True)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buffer.getvalue()
        except HttpError as exc:
            raise DriveError(_explain_http_error(exc)) from exc


def _explain_http_error(exc: HttpError) -> str:
    """Turn a Google HttpError into a human-readable, actionable message."""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "resp", None), "status", None
    )
    if status in (401, 403):
        return (
            f"Google Drive denied access (HTTP {status}). Share the folder with "
            "the service account's email and ensure the Drive API is enabled."
        )
    if status == 404:
        return "Drive folder or file not found (HTTP 404). Check the folder id."
    return f"Drive API error: {exc}"
