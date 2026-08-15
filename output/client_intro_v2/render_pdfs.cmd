@echo off
setlocal
rem Renders both intro HTML files to PDF using headless Edge.
rem Requires internet on first run (Google Fonts: Frank Ruhl Libre + Assistant).
set "EDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not exist "%EDGE%" set "EDGE=C:\Program Files\Microsoft\Edge\Application\msedge.exe"
set "D=%~dp0"
set "U=%D:\=/%"

echo Rendering Loren_Quant_Investing_Overview_2p.pdf ...
"%EDGE%" --headless --disable-gpu --no-pdf-header-footer --virtual-time-budget=12000 --print-to-pdf="%D%Loren_Quant_Investing_Overview_2p.pdf" "file:///%U%intro_2pager_he.html"

echo Rendering meeting_doc_he.pdf ...
"%EDGE%" --headless --disable-gpu --no-pdf-header-footer --virtual-time-budget=12000 --print-to-pdf="%D%meeting_doc_he.pdf" "file:///%U%meeting_doc_he.html"

echo.
echo Done. PDFs updated in this folder.
pause

