"""Reliable server-side PDF generation for finalized TripBandhu itineraries."""

from __future__ import annotations

import html
import os
import re
import unicodedata
from datetime import datetime
from io import BytesIO
from pathlib import Path
from threading import Lock

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


MAX_PDF_TEXT_LENGTH = 100_000
MAX_PDF_TITLE_LENGTH = 200

BRAND_GREEN = colors.HexColor("#064E3B")
BRAND_YELLOW = colors.HexColor("#F6C90E")
INK = colors.HexColor("#17201D")
MUTED = colors.HexColor("#5F6B66")
RULE = colors.HexColor("#D9DED8")
_FONT_REGISTRATION_LOCK = Lock()


class PdfContentError(ValueError):
    """Raised when a PDF request contains unusable or unsafe content."""


def _font_candidates() -> list[tuple[Path, Path]]:
    windows_dir = Path(os.environ.get("WINDIR", "C:/Windows"))
    reportlab_fonts = Path(__file__).resolve().parent
    try:
        import reportlab

        reportlab_fonts = Path(reportlab.__file__).resolve().parent / "fonts"
    except (ImportError, AttributeError):
        pass

    return [
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
        (
            reportlab_fonts / "Vera.ttf",
            reportlab_fonts / "VeraBd.ttf",
        ),
        (
            windows_dir / "Fonts" / "arial.ttf",
            windows_dir / "Fonts" / "arialbd.ttf",
        ),
    ]


def _register_fonts() -> tuple[str, str]:
    """Use Unicode-capable fonts where available, with a built-in fallback."""
    with _FONT_REGISTRATION_LOCK:
        registered_fonts = set(pdfmetrics.getRegisteredFontNames())
        if {"TripBandhuSans", "TripBandhuSans-Bold"} <= registered_fonts:
            return "TripBandhuSans", "TripBandhuSans-Bold"

        for regular_path, bold_path in _font_candidates():
            if regular_path.is_file() and bold_path.is_file():
                pdfmetrics.registerFont(TTFont("TripBandhuSans", str(regular_path)))
                pdfmetrics.registerFont(TTFont("TripBandhuSans-Bold", str(bold_path)))
                pdfmetrics.registerFontFamily(
                    "TripBandhuSans",
                    normal="TripBandhuSans",
                    bold="TripBandhuSans-Bold",
                )
                return "TripBandhuSans", "TripBandhuSans-Bold"

    return "Helvetica", "Helvetica-Bold"


def normalize_pdf_text(value: str) -> str:
    """Normalize model output into text that renders consistently in PDF fonts."""
    replacements = {
        "\u00a0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2022": "-",
        "\u00b7": "-",
        "\u2192": "->",
        "\u20b9": "INR ",
        "\u2713": "Yes",
        "\u2714": "Yes",
        "\u2717": "No",
        "\u2718": "No",
    }
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    for source, replacement in replacements.items():
        normalized = normalized.replace(source, replacement)

    # Emoji are decorative in generated itineraries and commonly render as boxes.
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) not in {"Cs", "Co"}
        and not (unicodedata.category(character) == "So" and ord(character) > 0x2600)
    )
    return normalized.replace("\x00", "").strip()


def _inline_markup(value: str) -> str:
    """Convert a safe Markdown subset into ReportLab Paragraph markup."""
    text = normalize_pdf_text(value)
    text = re.sub(r"\[([^\]]+)]\((https?://[^)]+)\)", r"\1 (\2)", text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*\s+", r"<b>\1</b>&#160;", text)
    text = re.sub(r"__(.+?)__\s+", r"<b>\1</b>&#160;", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font name=\"Courier\">\1</font>", text)
    return text


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _make_styles(font_name: str, bold_font_name: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TripBandhuTitle",
            parent=base["Normal"],
            fontName=bold_font_name,
            fontSize=20,
            leading=25,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "meta": ParagraphStyle(
            "TripBandhuMeta",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=9,
        ),
        "h1": ParagraphStyle(
            "TripBandhuH1",
            parent=base["Normal"],
            fontName=bold_font_name,
            fontSize=14,
            leading=18,
            textColor=BRAND_GREEN,
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "TripBandhuH2",
            parent=base["Normal"],
            fontName=bold_font_name,
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#0B6B50"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "TripBandhuBody",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=9.5,
            leading=14,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=4,
            splitLongWords=True,
        ),
        "bullet": ParagraphStyle(
            "TripBandhuBullet",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=9.5,
            leading=14,
            textColor=INK,
            leftIndent=13,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
        "quote": ParagraphStyle(
            "TripBandhuQuote",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=9,
            leading=13,
            textColor=MUTED,
            leftIndent=12,
            borderColor=BRAND_YELLOW,
            borderWidth=0,
            borderPadding=(0, 0, 0, 7),
            spaceAfter=5,
        ),
        "table": ParagraphStyle(
            "TripBandhuTable",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=7.5,
            leading=10,
            textColor=INK,
            splitLongWords=True,
        ),
        "table_header": ParagraphStyle(
            "TripBandhuTableHeader",
            parent=base["Normal"],
            fontName=bold_font_name,
            fontSize=7.5,
            leading=10,
            textColor=colors.white,
        ),
    }


def _append_markdown(
    story: list,
    markdown: str,
    styles: dict,
    document_width: float,
    document_title: str,
) -> None:
    lines = normalize_pdf_text(markdown).splitlines()
    index = 0
    first_level_one_heading_seen = False

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            story.append(Spacer(1, 2.5 * mm))
            index += 1
            continue

        if "|" in line and index + 1 < len(lines) and _is_table_separator(lines[index + 1]):
            table_rows = [_table_cells(line)]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_rows.append(_table_cells(lines[index]))
                index += 1

            column_count = max(len(row) for row in table_rows)
            normalized_rows = [row + [""] * (column_count - len(row)) for row in table_rows]
            rendered_rows = []
            for row_index, row in enumerate(normalized_rows):
                cell_style = styles["table_header"] if row_index == 0 else styles["table"]
                rendered_rows.append([Paragraph(_inline_markup(cell), cell_style) for cell in row])

            table = Table(
                rendered_rows,
                colWidths=[document_width / column_count] * column_count,
                repeatRows=1,
                hAlign="LEFT",
            )
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_GREEN),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([table, Spacer(1, 3 * mm)])
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2)
            duplicate_document_title = (
                level == 1
                and not first_level_one_heading_seen
                and normalize_pdf_text(heading_text).casefold() == document_title.casefold()
            )
            if level == 1:
                first_level_one_heading_seen = True
            if not duplicate_document_title:
                story.append(Paragraph(
                    _inline_markup(heading_text),
                    styles["h1" if level <= 2 else "h2"],
                ))
            index += 1
            continue

        bullet_match = re.match(r"^[-*+]\s+(.+)$", line)
        numbered_match = re.match(r"^(\d+)[.)]\s+(.+)$", line)
        if bullet_match:
            story.append(Paragraph(f"- {_inline_markup(bullet_match.group(1))}", styles["bullet"]))
        elif numbered_match:
            story.append(Paragraph(
                f"{numbered_match.group(1)}. {_inline_markup(numbered_match.group(2))}",
                styles["bullet"],
            ))
        elif line.startswith(">"):
            story.append(Paragraph(_inline_markup(line.lstrip("> ")), styles["quote"]))
        elif re.fullmatch(r"[-=_]{3,}", line):
            story.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceBefore=3, spaceAfter=5))
        else:
            story.append(Paragraph(_inline_markup(line), styles["body"]))
        index += 1


def _draw_page(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(BRAND_GREEN)
    canvas.rect(0, height - 5 * mm, width, 5 * mm, fill=1, stroke=0)
    canvas.setFillColor(BRAND_YELLOW)
    canvas.rect(0, height - 6.2 * mm, width, 1.2 * mm, fill=1, stroke=0)
    canvas.setStrokeColor(RULE)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8.5 * mm, "TripBandhu - AI Travel Research Assistant")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_itinerary_pdf(
    text: str,
    title: str = "TripBandhu Travel Plan",
    *,
    generated_at: datetime | None = None,
) -> bytes:
    """Build a polished A4 itinerary PDF and return its complete binary content."""
    normalized_text = normalize_pdf_text(text)
    normalized_title = normalize_pdf_text(title) or "TripBandhu Travel Plan"

    if not normalized_text:
        raise PdfContentError("A finalized travel plan is required before downloading a PDF.")
    if len(normalized_text) > MAX_PDF_TEXT_LENGTH:
        raise PdfContentError("The travel plan is too large to export as a PDF.")
    if len(normalized_title) > MAX_PDF_TITLE_LENGTH:
        raise PdfContentError("The PDF title is too long.")

    regular_font, bold_font = _register_fonts()
    styles = _make_styles(regular_font, bold_font)
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=normalized_title,
        author="TripBandhu",
        subject="AI-generated travel itinerary",
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        pageCompression=1,
    )

    timestamp = generated_at or datetime.now().astimezone()
    story = [
        Table(
            [[Paragraph("TRIPBANDHU", ParagraphStyle(
                "TripBandhuBrand",
                parent=styles["meta"],
                fontName=bold_font,
                fontSize=9,
                textColor=BRAND_GREEN,
                alignment=TA_CENTER,
            ))]],
            colWidths=[45 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), BRAND_YELLOW),
                ("BOX", (0, 0), (-1, -1), 0.8, BRAND_GREEN),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),
        Spacer(1, 4 * mm),
        Paragraph(_inline_markup(normalized_title), styles["title"]),
        Paragraph(
            f"Generated {timestamp.strftime('%d %b %Y, %I:%M %p %Z')}",
            styles["meta"],
        ),
        HRFlowable(width="100%", thickness=1, color=BRAND_GREEN, spaceAfter=5),
    ]
    _append_markdown(story, normalized_text, styles, document.width, normalized_title)

    document.build(story, onFirstPage=_draw_page, onLaterPages=_draw_page)
    pdf_bytes = buffer.getvalue()
    if not pdf_bytes.startswith(b"%PDF-"):
        raise RuntimeError("The PDF renderer returned an invalid document.")
    return pdf_bytes
