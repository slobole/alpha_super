# Vendored fonts

IBM Plex Sans and IBM Plex Mono, latin subset, WOFF2. Licensed under the
SIL Open Font License 1.1 (`LICENSE-IBMPlex.txt`), which permits embedding
and redistribution.

## Why these are in the repo rather than pulled from a CDN or the OS

The console and its saved report artifacts must render **identically on every
machine, with no network**. Both previous approaches failed that:

* A CDN link makes an artifact depend on being online to look right.
* OS fonts make it depend on *which OS*. The stack used to be
  `"Segoe UI", system-ui, …` / `"Cascadia Mono", Consolas, …`, so a report
  opened on a Mac fell through to SF Pro and Menlo, and on Linux to DejaVu.
  A report sent to an investor did not look like the report that was written.

Vendoring is what makes the artifact self-contained. Reports embed these files
as base64 `@font-face` rules; the Bench console serves them from `/fonts/` so
the browser can cache them instead of re-downloading per page.

## Files

| File | Faces covered |
|---|---|
| `IBMPlexSans-latin.woff2` | Variable, weight axis 100–700 — serves every prose weight |
| `IBMPlexMono-400-latin.woff2` | Regular |
| `IBMPlexMono-500-latin.woff2` | Medium — figure emphasis |

Latin subset only (~75 KB total). The reports contain no non-latin text; adding
the cyrillic/greek/vietnamese subsets Google serves would roughly quadruple the
per-artifact cost for glyphs nothing renders.

## Replacing or updating

Font stacks live in `alpha/engine/theme.py`, never in a stylesheet. Before
adding any face to a stack, verify it actually resolves in the browser that
renders the console — measure its text width against a deliberately nonsense
family name. Checking the OS font list is not sufficient: `Segoe UI Variable`
enumerates through GDI but is not addressable from Chromium, and was reverted
for exactly that reason.
