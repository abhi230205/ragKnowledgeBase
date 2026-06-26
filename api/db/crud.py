"""CRUD helpers over the config/files tables.

TODO (Phase 1/2): thin functions used by the routes and the ingestion pipeline:
    get_or_create_config(), update_config(), upsert_file_record(),
    delete_file_record(), list_file_records(), set_file_error().
Secret masking lives at the route layer, not here.
"""

from __future__ import annotations
