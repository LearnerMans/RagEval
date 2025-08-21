from __future__ import annotations

import logging
from typing import List

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_markdown


logger = logging.getLogger(__name__)


def _remove_noise_elements(soup: BeautifulSoup) -> None:
    for selector in [
        "script",
        "style",
        "noscript",
        "header",
        "nav",
        "footer",
        "aside",
        "form",
    ]:
        for element in soup.select(selector):
            element.decompose()

    noisy_keywords: List[str] = [
        "header",
        "footer",
        "nav",
        "navbar",
        "menu",
        "sidebar",
        "advert",
        "ad-",
        "ads",
        "promo",
        "subscribe",
        "cookie",
        "consent",
    ]

    for element in soup.find_all(True):
        element_id = (element.get("id") or "").lower()
        class_names = " ".join([c.lower() for c in (element.get("class") or [])])
        combined = f"{element_id} {class_names}"
        if any(keyword in combined for keyword in noisy_keywords):
            element.decompose()


def _extract_main_html(soup: BeautifulSoup) -> str:
    body = soup.body or soup
    main_tag = body.find("main")
    container = main_tag or body
    return str(container)


def extract_url_to_markdown(url: str, timeout_seconds: int = 20) -> str:
    """Fetch a URL, strip headers/navs/footers, and convert body HTML to Markdown."""
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        raise ValueError("URL did not return HTML content")

    soup = BeautifulSoup(response.text, "html.parser")
    _remove_noise_elements(soup)
    content_html = _extract_main_html(soup)

    markdown = html_to_markdown(
        content_html,
        heading_style="ATX",
        bullets="*",
        strip=['a'],
    )

    markdown = "\n".join(line.rstrip() for line in markdown.splitlines())
    return markdown.strip()


