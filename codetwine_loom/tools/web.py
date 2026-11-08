"""The two tools that reach outside the repository: a web search, and one page of the web.

web_search asks about something the repository does not contain ("do this the way <some
other project> does it now"). web_fetch reads the page of one web_search result. A URL
no web_search returned since reset_web_search_count() last ran is refused.

A call is sent to the broker when LLM_BROKER names one, and made in this process
otherwise (_search, _fetch).

A tool is off unless it has both a budget (WEB_SEARCH_MAX_CALL, WEB_FETCH_MAX_CALL) and
an endpoint (websearch.can_search(), websearch.can_fetch()), and web_fetch is off
without web_search (available(), fetch_available()).
"""

import functools
import logging
from collections.abc import Callable

from codetwine_loom import broker, config, textcut, websearch

logger = logging.getLogger(__name__)

# How many web_search and web_fetch calls have been spent since reset_web_search_count()
# and reset_web_fetch_count() last set them to 0
_web_search_count = 0
_web_fetch_count = 0

# The URLs web_search returned since reset_web_search_count() last cleared them, as
# websearch.url_key() gives them
_result_url_set: set[str] = set()


def reset_web_search_count() -> None:
    """Reset the web_search call counter to 0 and forget the URLs it returned."""
    global _web_search_count
    _web_search_count = 0
    _result_url_set.clear()
    websearch.forget_result_url()


def reset_web_fetch_count() -> None:
    """Reset the web_fetch call counter to 0."""
    global _web_fetch_count
    _web_fetch_count = 0


def available() -> bool:
    """Whether web_search is on for this run.

    Both the budget and the endpoint are asked here.

    Returns:
        True when there is a budget and an endpoint to spend it on.
    """
    return config.WEB_SEARCH_MAX_CALL > 0 and websearch.can_search()


def fetch_available() -> bool:
    """Whether web_fetch is on for this run.

    Returns:
        True when there is a budget and an endpoint to spend it on, and available() is
        True as well.
    """
    return config.WEB_FETCH_MAX_CALL > 0 and websearch.can_fetch() and available()


def call_remain_count() -> int:
    """Return how many web_search calls are left.

    Returns:
        WEB_SEARCH_MAX_CALL minus what has been used, never below 0.
    """
    return max(0, config.WEB_SEARCH_MAX_CALL - _web_search_count)


def fetch_call_remain_count() -> int:
    """Return how many web_fetch calls are left.

    Returns:
        WEB_FETCH_MAX_CALL minus what has been used, never below 0.
    """
    return max(0, config.WEB_FETCH_MAX_CALL - _web_fetch_count)


def web_search_count() -> int:
    """Return how many web_search calls have been spent since the last reset.

    Returns:
        The number of calls made since the last reset.
    """
    return _web_search_count


def web_fetch_count() -> int:
    """Return how many web_fetch calls have been spent since the last reset.

    Returns:
        The number of calls made since the last reset.
    """
    return _web_fetch_count


def _within_limit(result_list: list[dict]) -> list[dict]:
    """Cut the results to what fits in WEB_SEARCH_RESULT_MAX_CHAR.

    The whole list is measured: the cap holds over everything that comes back, not over
    one result.

    Args:
        result_list: The results as the endpoint returned them.

    Returns:
        As many of the leading results as fit in WEB_SEARCH_RESULT_MAX_CHAR, and the
        first one alone when even that does not fit. All of them when
        WEB_SEARCH_RESULT_MAX_CHAR is 0.
    """
    keep_list: list[dict] = []
    for result_dict in result_list:
        if textcut.json_over_count(keep_list + [result_dict],
                                   config.WEB_SEARCH_RESULT_MAX_CHAR) > 0 and keep_list:
            logger.info("the results are long. Keeping the first %d of %d",
                        len(keep_list), len(result_list))
            return keep_list
        keep_list.append(result_dict)
    return keep_list


def _refuse(error: str, calls_left: int) -> dict:
    """Return one refusal of a tool, with its budget on it.

    Args:
        error: What went wrong.
        calls_left: How many calls of the tool are left.

    Returns:
        {"ok": False, "error": str, "calls_left": int}.
    """
    return {"ok": False, "error": error, "calls_left": calls_left}


# What web_search says to do in place of a search that did not run. {other} is
# _OTHER_WAY or _OTHER_WAY_NO_RUN (search_tool)
_NO_SEARCH_ERROR = "the search did not run ({kind}). Settle this from {other} instead"
_OTHER_WAY = "the repository or from run_project"
_OTHER_WAY_NO_RUN = "the repository"


def web_search(question: str) -> dict:
    """Ask the web one question about something outside this repository.

    The question is sent to the search endpoint as written and the results come back as
    they are: "title", "url" and "content" (an excerpt) each, as far as the endpoint
    supplies them. Nothing reads them for you.

    A question over WEB_SEARCH_QUESTION_MAX_CHAR is refused without spending a call and
    without leaving this machine.

    Args:
        question: What to find out, as one short question in English. It leaves this
            machine as written and carries no repository content.

    Returns:
        {"ok": True, "question": str, "results": list[dict], "truncated": bool,
         "calls_left": int}, or
        {"ok": False, "error": str, "calls_left": int} when the question is not a
        string, the tool is off, the budget is spent, the question is too long, there is
        no endpoint, the search failed, or it found nothing. "calls_left" is there
        either way, and "truncated" is True when results past WEB_SEARCH_RESULT_MAX_CHAR
        were left out.

    Raises:
        ValueError: When question is empty or whitespace only.
    """
    return _ask(question, _OTHER_WAY)


def search_tool(with_run_project: bool) -> Callable[[str], dict]:
    """Return web_search, or a stand-in whose failure does not name run_project.

    Args:
        with_run_project: Whether the answer for a search that did not run may name
            run_project.

    Returns:
        web_search itself, or otherwise a function with its name, signature and
        docstring whose answer for a search that did not run does not name run_project.
    """
    if with_run_project:
        return web_search

    @functools.wraps(web_search)
    def search_no_run(question: str) -> dict:
        """Make one web_search call whose failure does not name run_project."""
        return _ask(question, _OTHER_WAY_NO_RUN)
    return search_no_run


def _ask(question: str, other_way: str) -> dict:
    """Make one web_search call.

    Args:
        question: What to find out, as web_search takes it.
        other_way: What the refusal of a search that did not run names in its place
            (_OTHER_WAY or _OTHER_WAY_NO_RUN).

    Returns:
        What web_search returns.

    Raises:
        ValueError: When question is empty or whitespace only.
    """
    global _web_search_count
    type_error = textcut.text_error(question=question)
    if type_error:
        return _refuse(type_error, call_remain_count())
    if not question or not question.strip():
        raise ValueError("empty question. Pass one short question in English")
    if not available():
        return _refuse("web_search is turned off for this run", call_remain_count())
    if _web_search_count >= config.WEB_SEARCH_MAX_CALL:
        logger.info("web_search hit its call limit: %d calls", _web_search_count)
        return _refuse(f"web_search hit its call limit "
                       f"({config.WEB_SEARCH_MAX_CALL} calls)", call_remain_count())
    refusal = websearch.check_question(question)
    if refusal:
        return _refuse(refusal, call_remain_count())

    _web_search_count += 1
    try:
        result_list = _search(question.strip())
    # A search that did not happen comes back as ok=False rather than as an empty
    # result. The reason goes to the log
    except (ValueError, RuntimeError) as e:
        logger.warning("the search did not run: %s: %s", type(e).__name__,
                       str(e)[-config.LOG_EXCERPT_CHAR:])
        return _refuse(_NO_SEARCH_ERROR.format(kind=type(e).__name__, other=other_way),
                       call_remain_count())
    if not result_list:
        return _refuse("the search found nothing for that question", call_remain_count())
    keep_list = _within_limit(result_list)
    _result_url_set.update(key for result_dict in keep_list
                           if isinstance(result_dict.get("url"), str)
                           and (key := websearch.url_key(result_dict["url"])))
    return {"ok": True, "question": question.strip(), "results": keep_list,
            "truncated": len(keep_list) < len(result_list),
            "calls_left": call_remain_count()}


def _search(question: str) -> list[dict]:
    """Make the search through the broker when LLM_BROKER is set, and here otherwise.

    Args:
        question: The question, already checked.

    Returns:
        The results the endpoint returned.

    Raises:
        ValueError: Made here, when the question may not be sent, there is no endpoint,
            or the reply holds no result list (websearch.search); through the broker,
            when config.LLM_BROKER is not an address broker.endpoint() understands.
        RuntimeError: When the search failed, and through the broker when the broker
            cannot be reached, refuses the question or has no endpoint (broker.search).
    """
    if config.LLM_BROKER:
        return broker.search(question)
    return websearch.search(question)


def web_fetch(url: str) -> dict:
    """Read one page of the web by its URL.

    The URL goes to the page-reading endpoint as the search returned it, without its
    fragment (websearch.fetch), and the page's text comes back as the endpoint
    extracted it, cut to WEB_FETCH_RESULT_MAX_CHAR. Nothing reads it for you.

    A URL no web_search returned since reset_web_search_count() last ran is refused
    without spending a call and without leaving this machine.

    Args:
        url: The "url" of one web_search result.

    Returns:
        {"ok": True, "url": str, "content": str, "truncated": bool, "calls_left": int},
        or {"ok": False, "error": str, "calls_left": int} when the URL is not a
        string, the tool is off, the budget is spent, no web_search returned the URL
        (one that cannot be read as a URL included), there is no endpoint, or the read
        failed. "calls_left" is there either way, and "truncated" is True when the
        text was cut at WEB_FETCH_RESULT_MAX_CHAR.

    Raises:
        ValueError: When url is empty or whitespace only.
    """
    global _web_fetch_count
    type_error = textcut.text_error(url=url)
    if type_error:
        return _refuse(type_error, fetch_call_remain_count())
    if not url or not url.strip():
        raise ValueError("empty URL. Pass the \"url\" of one web_search result")
    if not fetch_available():
        return _refuse("web_fetch is turned off for this run", fetch_call_remain_count())
    if _web_fetch_count >= config.WEB_FETCH_MAX_CALL:
        logger.info("web_fetch hit its call limit: %d calls", _web_fetch_count)
        return _refuse(f"web_fetch hit its call limit "
                       f"({config.WEB_FETCH_MAX_CALL} calls)", fetch_call_remain_count())
    if websearch.url_key(url) not in _result_url_set:
        return _refuse(websearch.UNKNOWN_URL_ERROR, fetch_call_remain_count())

    _web_fetch_count += 1
    try:
        content = _fetch(url.strip())
    # A page that was not read comes back as ok=False. The reason goes to the log
    except (ValueError, RuntimeError) as e:
        logger.warning("the page was not read: %s: %s", type(e).__name__,
                       str(e)[-config.LOG_EXCERPT_CHAR:])
        return _refuse(f"the page was not read ({type(e).__name__}). Try another "
                       f"result, or settle this from the repository instead",
                       fetch_call_remain_count())
    keep_content = textcut.cut(content, config.WEB_FETCH_RESULT_MAX_CHAR)
    return {"ok": True, "url": url.strip(), "content": keep_content,
            "truncated": keep_content != content,
            "calls_left": fetch_call_remain_count()}


def _fetch(url: str) -> str:
    """Read the page through the broker when LLM_BROKER is set, and here otherwise.

    Args:
        url: The page's URL, already checked.

    Returns:
        The page's text, whole.

    Raises:
        ValueError: Made here, when no search returned the URL, there is no endpoint, or
            the reply holds no page or no text (websearch.fetch); through the broker,
            when config.LLM_BROKER is not an address broker.endpoint() understands.
        RuntimeError: When the read failed, and through the broker when the broker
            cannot be reached, refuses the URL or has no endpoint (broker.fetch).
    """
    if config.LLM_BROKER:
        return broker.fetch(url)
    return websearch.fetch(url)
