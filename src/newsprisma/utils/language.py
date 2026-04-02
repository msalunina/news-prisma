"""Language detection — wraps langdetect and normalises to 'en' | 'es' | 'ru'."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SUPPORTED = {"en", "es", "ru"}
_FALLBACK = "en"


def detect_language(text: str) -> str:
    """Detect the language of *text* and return 'en', 'es', or 'ru'.

    Falls back to 'en' if the language cannot be determined or is not one of
    the three supported languages.
    """
    from langdetect import DetectorFactory, LangDetectException, detect

    # Make detection deterministic across runs
    DetectorFactory.seed = 0

    if not text or len(text.strip()) < 10:
        return _FALLBACK

    try:
        lang = detect(text)
        if lang in _SUPPORTED:
            return lang
        logger.debug("Detected unsupported language %r — falling back to %r", lang, _FALLBACK)
        return _FALLBACK
    except LangDetectException:
        logger.debug("Language detection failed — falling back to %r", _FALLBACK)
        return _FALLBACK
