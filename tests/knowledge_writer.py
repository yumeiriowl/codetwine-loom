"""Write a consolidated analysis dict out as the knowledge database the tool reads.

Only what the index build reads is written: the file entries with their dependencies
and design documents.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

# The table layout comes from the module that reads the database
from codetwine.knowledge_db import SCHEMA_VERSION, _SCHEMA


def write_knowledge_db(knowledge_dict: dict, path: str, project_name: str = "") -> str:
    """Write one consolidated analysis dict out as a knowledge database.

    An existing database at that path is replaced.

    Args:
        knowledge_dict: A consolidated analysis, i.e. {"files": [{"file", "file_dependencies",
            "doc"}, ...]}. "project_dependencies" is used for the per-file summaries
            and the file-to-file edges when it is there.
        path: Where to write the database.
        project_name: The name to record in meta. Falls back to pk["project_name"].

    Returns:
        The path that was written.
    """
    if os.path.exists(path):
        os.remove(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    summary_dict, edge_list = _dependency_row_tuple(knowledge_dict)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(_SCHEMA)
        connection.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [("project_name", project_name or knowledge_dict.get("project_name", "")),
             ("schema_version", SCHEMA_VERSION),
             ("created_at", datetime.now(timezone.utc).isoformat())])
        for entry_dict in knowledge_dict.get("files", []):
            _insert_file(connection, entry_dict, summary_dict)
        connection.executemany(
            "INSERT OR IGNORE INTO file_edges (file, direction, other) "
            "VALUES (?, ?, ?)", edge_list)
        connection.commit()
    finally:
        connection.close()
    return path


def _dependency_row_tuple(knowledge_dict: dict) -> tuple[dict, list[tuple[str, str, str]]]:
    """Take the per-file summaries and the file-to-file edges out of the analysis.

    Args:
        knowledge_dict: A consolidated analysis.

    Returns:
        ({file path: summary}, [(file, "caller" or "callee", the other file), ...]).
    """
    summary_dict, edge_list = {}, []
    for entry_dict in knowledge_dict.get("project_dependencies", []):
        summary_dict[entry_dict["file"]] = entry_dict.get("summary")
        for direction in ("caller", "callee"):
            for other in entry_dict.get(direction + "s", []):
                edge_list.append((entry_dict["file"], direction, other))
    return summary_dict, edge_list


def _insert_file(connection: sqlite3.Connection, entry_dict: dict,
                 summary_dict: dict) -> None:
    """Insert one file entry and its definitions.

    Args:
        connection: An open connection to the database being written.
        entry_dict: One element of the analysis's "files".
        summary_dict: {file path: summary}, as _dependency_row_tuple() returned it.
    """
    dependency_dict = entry_dict.get("file_dependencies")
    doc = entry_dict.get("doc")
    summary = summary_dict.get(entry_dict["file"])
    if summary is None and doc:
        summary = doc.get("summary")
    connection.execute(
        "INSERT INTO files (file, summary, file_dependencies, doc) VALUES (?, ?, ?, ?)",
        (entry_dict["file"], summary,
         json.dumps(dependency_dict, ensure_ascii=False) if dependency_dict else None,
         json.dumps(doc, ensure_ascii=False) if doc else None))
    connection.executemany(
        "INSERT INTO definitions (file, name, type, start_line, end_line) "
        "VALUES (?, ?, ?, ?, ?)",
        [(entry_dict["file"], d.get("name"), d.get("type"),
          d.get("start_line"), d.get("end_line"))
         for d in (dependency_dict or {}).get("definitions", [])])
