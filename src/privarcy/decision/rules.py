"""Regex-based PII detection rules for OCR'd text.

Deliberately simple, readable patterns rather than a full PII-detection
library — good enough to catch clearly-structured PII (card numbers, SSNs,
emails, phone numbers) in OCR output, with the understanding that OCR
text is noisy (misread characters, missing spaces) so these are
intentionally a bit permissive rather than pixel-perfect format validators.
"""

import re
from typing import List, Tuple

PII_PATTERNS = {
    # 13-19 digits, optionally grouped by spaces/dashes — covers most card
    # network formats without validating a real Luhn checksum (OCR text
    # is too noisy for that to be reliable).
    "credit_card_number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    # US Social Security Number format: 123-45-6789
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
    # Loosely matches US-style phone numbers with optional country code.
    "phone_number": re.compile(r"\b(?:\+?1[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"),
    # Passport-like: 1-2 letters followed by 6-9 digits.
    "passport_like": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
    "date_of_birth": re.compile(
        r"\b(?:DOB|D\.O\.B\.|Date\s+of\s+Birth)\s*[:\-]?\s*"
        r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    "name": re.compile(
        r"\b(?:Full\s+Name|Name)\s*[:\-]\s*"
        r"[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3}\b",
        re.IGNORECASE,
    ),
    "address": re.compile(
        r"\b(?:(?:Home|Residential|Mailing)\s+)?Address\s*[:\-]\s*[^\r\n]{5,100}",
        re.IGNORECASE,
    ),
    "government_id": re.compile(
        r"\b(?:Government\s+ID|National\s+ID|ID\s+Number|Document\s+No\.?)\s*"
        r"[:\-]\s*[A-Z0-9][A-Z0-9 -]{4,24}\b",
        re.IGNORECASE,
    ),
    "account_number": re.compile(
        r"\b(?:Account\s+Number|Account\s+No\.?|Bank\s+Account)\s*"
        r"[:\-]\s*[0-9 -]{6,24}\b",
        re.IGNORECASE,
    ),
    "certificate_or_registration_number": re.compile(
        r"\b(?:Certificate|Registration)\s+(?:No\.?|Number)\s*"
        r"[:\-]\s*[A-Z0-9][A-Z0-9 -]{3,24}\b",
        re.IGNORECASE,
    ),
}


def find_sensitive_spans(text: str) -> List[dict]:
    """Return matching sensitive substrings and their character offsets."""
    if not text:
        return []

    spans = [
        {"pattern": name, "start": match.start(), "end": match.end(), "text": match.group()}
        for name, pattern in PII_PATTERNS.items()
        for match in pattern.finditer(text)
    ]
    return sorted(spans, key=lambda span: (span["start"], span["end"]))


def contains_pii(text: str) -> Tuple[bool, List[str]]:
    """Check OCR'd text against every PII pattern.

    Returns:
        (True/False whether ANY pattern matched, list of pattern names that matched).
    """
    if not text:
        return False, []

    matched = [name for name, pattern in PII_PATTERNS.items() if pattern.search(text)]
    return (len(matched) > 0), matched
