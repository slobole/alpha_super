"""Prepare the monthly report's pictures: crop to the report's band and save small.

Usage:
    python scripts/prepare_report_art.py <folder or file> [<folder or file> ...] [--month YYYY-MM]

Source files are named by month (2026-11.png). Any PNG, JPG or WebP works; --month names
a single file that has another name. Output: static/report_art/<YYYY-MM>.jpg, which the
report finds by itself. A whole year can be prepared at once, so no monthly step is needed.
"""

import argparse
from pathlib import Path
import re
import sys

from PIL import Image, ImageOps


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
ART_DIRECTORY_PATH = REPO_ROOT_PATH / "alpha" / "live" / "dashboard_v4" / "static" / "report_art"
# Twice the report's band (794 x 176 CSS px), the crop the page would otherwise make itself.
BAND_WIDTH_INT, BAND_HEIGHT_INT = 1588, 352
BAND_VERTICAL_POSITION_FLOAT = .47
MINIMUM_SOURCE_WIDTH_INT = 1200
JPEG_QUALITY_INT = 85
MONTH_PATTERN_STR = r"20[0-9]{2}-(0[1-9]|1[0-2])"
SOURCE_SUFFIX_SET = {".png", ".jpg", ".jpeg", ".webp"}


class ReportArtError(ValueError):
    """A source picture cannot become a report picture."""


def prepare_picture_path(source_path, month_str, *, output_directory_path=ART_DIRECTORY_PATH):
    """Save one month's band and return its path. Nothing is written for a bad source."""
    if not re.fullmatch(MONTH_PATTERN_STR, month_str):
        raise ReportArtError(f"{source_path.name}: the month must look like 2026-11.")
    try:
        with Image.open(source_path) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
    except OSError as error_obj:
        raise ReportArtError(f"{source_path.name}: not a readable picture.") from error_obj
    width_int, height_int = image.size
    if width_int < MINIMUM_SOURCE_WIDTH_INT:
        raise ReportArtError(f"{source_path.name}: {width_int} pixels wide, at least {MINIMUM_SOURCE_WIDTH_INT} are needed.")
    crop_height_int = round(width_int * BAND_HEIGHT_INT / BAND_WIDTH_INT)
    if crop_height_int > height_int:
        raise ReportArtError(f"{source_path.name}: too wide and flat for the band; use a 16:9 picture.")
    top_int = round((height_int - crop_height_int) * BAND_VERTICAL_POSITION_FLOAT)
    band = image.crop((0, top_int, width_int, top_int + crop_height_int)).resize((BAND_WIDTH_INT, BAND_HEIGHT_INT), Image.LANCZOS)
    output_directory_path.mkdir(parents=True, exist_ok=True)
    output_path = output_directory_path / f"{month_str}.jpg"
    band.save(output_path, "JPEG", quality=JPEG_QUALITY_INT, optimize=True, progressive=True)
    return output_path


def source_month_list(argument_list, month_str):
    """Pairs of (file, month). A folder contributes its files named by month and nothing else."""
    pair_list = []
    for argument_str in argument_list:
        path = Path(argument_str)
        if path.is_dir():
            pair_list += [(child, child.stem) for child in sorted(path.iterdir())
                if child.suffix.lower() in SOURCE_SUFFIX_SET and re.fullmatch(MONTH_PATTERN_STR, child.stem)]
        else:
            pair_list.append((path, month_str or path.stem))
    if month_str and len(pair_list) != 1:
        raise ReportArtError("--month names exactly one file.")
    return pair_list


def main(argument_list=None):
    parser = argparse.ArgumentParser(description="Prepare monthly report pictures.")
    parser.add_argument("source", nargs="+", help="A folder of pictures named by month, or picture files.")
    parser.add_argument("--month", help="The month of a single file with another name, like 2026-11.")
    argument_obj = parser.parse_args(argument_list)
    try:
        pair_list = source_month_list(argument_obj.source, argument_obj.month)
        if not pair_list:
            raise ReportArtError("No picture named by month was found.")
        for source_path, month_str in pair_list:
            output_path = prepare_picture_path(source_path, month_str)
            print(f"{month_str}: {output_path.stat().st_size // 1000} KB -> {output_path}")
    except ReportArtError as error_obj:
        print(f"Not prepared. {error_obj}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
