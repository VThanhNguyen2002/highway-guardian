"""
backend/core/pdf_parser.py

PDF-based traffic sign rule validator.
Parses the QCVN 41:2019 PDF at startup and exposes an O(1) validation lookup.
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber


# Regex matching Vietnamese traffic sign codes: P.102, W.201a, R.301c, S.509a, etc.
_SIGN_CODE_PATTERN: re.Pattern[str] = re.compile(
    r"\b[PRWS]\.\d{3}[a-z]?\b"
)


class PDFRuleParser:
    """Extracts and indexes traffic sign codes from a QCVN 41:2019 PDF.

    The PDF is parsed exactly once at instantiation. All subsequent validity
    checks are O(1) frozenset lookups.

    Args:
        pdf_path: Absolute path to the QCVN 41:2019 standard PDF.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        RuntimeError: If ``pdfplumber`` cannot open or parse the PDF.

    Example:
        >>> parser = PDFRuleParser(Path("/app/docs/qcvn41.pdf"))
        >>> parser.is_sign_valid("P.102")
        True
    """

    def __init__(self, pdf_path: Path) -> None:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"[PDFRuleParser] Parsing PDF: {pdf_path}")
        self._valid_codes: frozenset[str] = self._extract_sign_codes(pdf_path)
        print(
            f"[PDFRuleParser] Indexed {len(self._valid_codes)} unique sign codes "
            f"from PDF."
        )

    @staticmethod
    def _extract_sign_codes(pdf_path: Path) -> frozenset[str]:
        """Extract all traffic sign code strings from every page of the PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Immutable set of uppercase sign code strings (e.g. ``{"P.102", "W.201a"}``).
        """
        codes: set[str] = set()
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    text: str = page.extract_text() or ""
                    matches = _SIGN_CODE_PATTERN.findall(text)
                    codes.update(matches)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse PDF at {pdf_path}: {exc}"
            ) from exc

        return frozenset(codes)

    def is_sign_valid(self, sign_code: str) -> bool:
        """Check whether a sign code appears in the QCVN 41:2019 standard.

        Args:
            sign_code: A traffic sign code string (e.g. ``"P.102"``).

        Returns:
            True if the code is present in the PDF rule corpus; False otherwise.
        """
        return sign_code in self._valid_codes

    @property
    def indexed_codes(self) -> frozenset[str]:
        """Return a read-only view of all indexed sign codes.

        Returns:
            Frozenset of all sign codes extracted from the PDF.
        """
        return self._valid_codes
