@echo off
rem Double-click to start kinesics in the background (no console window).
rem It starts armed; ctrl+alt+G disarms, ctrl+alt+Q quits.
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0gesture_scroll.py"
