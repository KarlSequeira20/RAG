"""Fetch a web page and turn it into clean text.

The hard part of web ingestion is not fetching, it is deciding what on the page is
actually the content. A typical page is mostly navigation, cookie banners, footers,
"related articles" and ads. Indexed naively, those become chunks that compete with real
content during retrieval -- the same problem the hidden LaTeX source caused in the PDFs.

`trafilatura` exists for exactly this: it identifies the main article body and discards
the furniture around it.
"""

import trafilatura

# Sent so the request is identifiable rather than anonymous.
USER_AGENT = "rag-papers-demo/1.0 (personal learning project)"


class FetchError(Exception):
    """Raised when a URL cannot be fetched or holds no usable text."""


def fetch_page(url: str):
    """Download a URL and extract its main text. Returns (title, text).

    Raises FetchError with a readable message rather than letting network or parsing
    failures surface as tracebacks, since this is called from the UI.
    """
    if not url.startswith(("http://", "https://")):
        raise FetchError("URL must start with http:// or https://")

    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise FetchError("Could not fetch that URL. It may be unreachable or blocking requests.")

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not text or not text.strip():
        raise FetchError(
            "No readable text found on that page. Pages built entirely in JavaScript "
            "often return nothing, because only the initial HTML is fetched -- no "
            "browser runs to build the page."
        )

    metadata = trafilatura.extract_metadata(downloaded)
    title = (metadata.title if metadata and metadata.title else url).strip()
    return title, text
