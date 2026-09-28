"""C3/C4: listing venue, security name and first quoted date for every traded symbol (Norgate metadata).

Usage: uv run python tb_venues.py
"""

from __future__ import annotations

import tb_common as tc
from data.norgate_loader import norgatedata

SYMBOLS = ("SPY", "TLT", "IEF", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "VOX", "IYR",
           "SOXX", "IGV", "IBB", "KIE", "IHI", "XLC")


def main() -> None:
    out = {}
    for s in SYMBOLS:
        rec = {}
        for fn in ("exchange_name", "security_name", "first_quoted_date", "last_quoted_date", "subtype1"):
            try:
                rec[fn] = str(getattr(norgatedata, fn)(s))
            except Exception as exc:  # metadata call not available in this package version
                rec[fn] = f"n/a ({type(exc).__name__})"
        out[s] = rec
        print(s, rec, flush=True)
    tc.write_json("venues.json", out)


if __name__ == "__main__":
    main()
