"""The entry point of index building: steps 1-5.

build_index() is steps 1-4; build_and_save() adds step 5, writing the index out.
"""

import logging

from codetwine_loom import knowledge as knowledge_mod
from codetwine_loom.index import store
from codetwine_loom.index.docs import attach_doc, file_doc
from codetwine_loom.index.graph import (add_def_node, add_hold_name, build_call_list,
                                        def_by_file_dict, key_by_file_name_dict,
                                        mark_record)
from codetwine_loom.index.indexer import build_bm25
from codetwine_loom.knowledge import Knowledge
from codetwine_loom.languages import file_lang, set_file_lang
from codetwine_loom.paths import knowledge_to_repo

logger = logging.getLogger(__name__)


def path_term_text(knowledge_path: str, project_name: str) -> str:
    """Turn a path into search terms: separators become spaces.

    A knowledge path is converted to a repository-relative path first
    (knowledge_to_repo()), which drops its <stem>_<ext> directory.

    Examples:
        "myapp/myapp/config/settings_py/settings.py" -> "myapp config settings py "

    Args:
        knowledge_path: A knowledge path.
        project_name: The directory name of PROJECT_DIR.

    Returns:
        The space-separated words, ending with one space.
    """
    repo_path = knowledge_to_repo(knowledge_path, project_name)
    return repo_path.replace("/", " ").replace(".", " ") + " "


def bm25_doc_dict(def_dict: dict, project_name: str) -> dict:
    """Build the searchable text per definition for the definition index.

    The code and the prose are kept apart.

    Args:
        def_dict: The definition nodes after steps 1-3.
        project_name: The directory name of PROJECT_DIR, used for the path words.

    Returns:
        {definition key: ("signature", "doc", the names the definition holds
        (graph.add_hold_name()), "path words")}.
    """
    return {key: (node_dict["signature"], node_dict["doc"],
                  node_dict.get("hold_name_list", []),
                  path_term_text(node_dict["file"], project_name))
            for key, node_dict in def_dict.items()}


def file_bm25_doc_dict(file_dict: dict, project_name: str) -> dict:
    """Build the searchable text per file for the file index.

    A file with neither a summary nor a definition is left out.

    Args:
        file_dict: The file nodes of step 1.
        project_name: The directory name of PROJECT_DIR, used for the path words.

    Returns:
        {knowledge path: ("", "file summary", [], "path words")}.
    """
    return {path: ("", node_dict["summary"], [], path_term_text(path, project_name))
            for path, node_dict in file_dict.items()
            if node_dict["summary"] or node_dict["defs"]}


def build_index(knowledge: Knowledge, project_name: str) -> dict:
    """Call steps 1-4 in order and assemble what they build into one dict.

    Steps:
    1. Build the definition nodes and the file nodes (step 1)
    2. Build the calls edges (step 2)
    3. Attach the design docs to the definitions (step 3)
    4. Build the two BM25 inverted indexes (step 4)

    The analysis is walked once. What the later steps need - the references and the
    design documents - is taken out of each file entry as it goes past; a definition's
    code is not kept, only the one line of it the signature is. attach_doc() rewrites
    defs in place and runs before build_bm25.

    Args:
        knowledge: The analysis to build the index from.
        project_name: The directory name of PROJECT_DIR, used for the path words.

    Returns:
        {"nodes": {"defs": dict, "files": dict},
         "edges": {"calls": list},
         "index": {"def": dict, "file": dict},
         "file_lang": {knowledge path: languages.file_lang() of it}}, the last only
        for the files where file_lang() is not "". The files of file_lang are also
        handed to languages.set_file_lang().
    """
    # An analysis of 0 files is logged as a warning and the build goes on
    if not knowledge.file_count():
        logger.warning("the analysis covers 0 files; building an empty index")

    # == Step 1: definition nodes and file nodes =============
    def_dict: dict = {}
    file_dict: dict = {}
    usage_dict: dict = {}
    doc_dict: dict = {}
    lang_dict: dict = {}
    for file_entry in knowledge.iter_file():
        path = file_entry["file"]
        dependency_dict = file_entry["file_dependencies"]
        add_def_node(def_dict, file_entry)
        if lang := file_lang(file_entry):
            lang_dict[path] = lang
        doc = file_doc(file_entry)
        file_dict[path] = {"summary": doc["summary"],
                           "doc_hash": doc["source_hash"], "defs": []}
        # A use of the file's own definition is a reference from the file to itself
        usage_dict[path] = dependency_dict["callee_usages"] + [
            {**usage, "from": path} for usage in dependency_dict["same_file_usages"]]
        doc_dict[path] = doc
    set_file_lang(lang_dict)
    by_file_dict = def_by_file_dict(def_dict)
    by_file_name_dict = key_by_file_name_dict(def_dict)
    mark_record(def_dict, by_file_dict)
    add_hold_name(def_dict, by_file_dict)
    for path, node_dict in file_dict.items():
        node_dict["defs"] = [key for _, _, key in by_file_dict.get(path, ())]

    # == Step 2: calls edges ==================================
    call_list = build_call_list(usage_dict, by_file_name_dict, by_file_dict)

    # == Step 3: attach the design docs ======================
    attach_doc(doc_dict, def_dict, by_file_name_dict, by_file_dict)

    # == Step 4: the BM25 indexes (definitions and files) ====
    return {
        "nodes": {"defs": def_dict, "files": file_dict},
        "edges": {"calls": call_list},
        "index": {"def": build_bm25(bm25_doc_dict(def_dict, project_name)),
                  "file": build_bm25(file_bm25_doc_dict(file_dict, project_name))},
        "file_lang": lang_dict,
    }


def commit_of_analysis(index_path: str, source_stamp: str) -> str:
    """Return the commit the analysis on disk describes, as what is beside it says.

    The record beside the analysis (knowledge.ANALYSIS_COMMIT_FILE) is read first; when
    there is no such record for the analysis or it names no commit, an index already at
    index_path built from that same analysis is what says it. The analysis and the index
    are looked for in the same directory.

    Args:
        index_path: The path to code_index.sqlite.
        source_stamp: The analysis the answer has to belong to (Knowledge.stamp()).

    Returns:
        The full commit hash, or "" when neither can say.
    """
    knowledge_path = knowledge_mod.knowledge_db_path_of_index(index_path)
    return (knowledge_mod.analysis_commit(knowledge_path, source_stamp)
            or store.commit_in_index(index_path, source_stamp))


def is_index_from(knowledge_path: str, index_path: str) -> bool:
    """Say whether the index at index_path was built from the analysis at knowledge_path.

    Args:
        knowledge_path: The analysis.
        index_path: Where code_index.sqlite is.

    Returns:
        store.is_from_analysis() for the analysis's stamp.
    """
    knowledge = Knowledge.open(knowledge_path)
    try:
        return store.is_from_analysis(index_path, knowledge.stamp())
    finally:
        knowledge.close()


def index_again(knowledge_path: str, project_name: str, index_path: str) -> None:
    """Build the index of the analysis at knowledge_path and write it out (no LLM).

    Args:
        knowledge_path: The analysis.
        project_name: The directory name of PROJECT_DIR.
        index_path: Where code_index.sqlite is written.
    """
    knowledge = Knowledge.open(knowledge_path)
    try:
        build_and_save(knowledge, project_name, index_path)
    finally:
        knowledge.close()


def build_and_save(knowledge: Knowledge, project_name: str, index_path: str) -> dict:
    """Build the index (steps 1-4) and write it out (step 5).

    The commit the index records is the one its *analysis* describes. It is the commit
    the record beside the analysis names, the one checked out when the working tree did
    not differ from it; when there is no such record for the analysis or it names none,
    what the index already there said about that same analysis; and "" when neither can
    say.

    Args:
        knowledge: The analysis to build the index from.
        project_name: The directory name of PROJECT_DIR.
        index_path: Where code_index.sqlite is written.

    Returns:
        build_index()'s return value.
    """
    source_stamp = knowledge.stamp()
    index_dict = build_index(knowledge, project_name)
    store.save(index_path, index_dict, index_dict["nodes"]["defs"], source_stamp, project_name,
               commit_of_analysis(index_path, source_stamp))
    return index_dict


def analyse_and_index(project_dir: str, output_dir: str,
                      exclude_list: list[str] | None = None,
                      with_doc: bool = False) -> str:
    """Make the analysis in output_dir again over the working tree and index it.

    Without the documents no LLM is called, the design documents already there stay, and
    an analysis that already describes the tree is used as it is (ensure_knowledge());
    with them codetwine writes the documents for the files that changed and what depends
    on them. The definitions and the line numbers are made again whenever codetwine
    runs.

    Args:
        project_dir: The path to the target repository.
        output_dir: Where the analysis and the index live.
        exclude_list: Extra directory and file names kept out of the analysis.
        with_doc: Whether codetwine generates the design documents too.

    Returns:
        The path to the knowledge database, as ensure_knowledge() returns it.

    Raises:
        ValueError: When project_dir does not exist or is not a directory.
        RuntimeError: When it is not a git repository, or codetwine left no database.
    """
    path, _, project_name, _ = knowledge_mod.ensure_knowledge(
        project_dir, output_dir, rebuild=True, with_doc=with_doc,
        exclude_list=exclude_list)
    index_again(path, project_name, store.index_db_path(output_dir, project_name))
    return path
