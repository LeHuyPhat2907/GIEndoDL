@echo off
title GIEndoDL - He Thong CADx Noi Soi Tieu Hoa
chcp 65001 > nul
cd /d "%~dp0"

echo ======================================================================
echo    GIEndoDL - HE THONG AI HO TRO CHAN DOAN NOI SOI TIEU HOA
echo    Mo hinh: Hybrid CNN-CBAM-Transformer (Run 4 Production - 19 Lop)
echo ======================================================================
echo.
echo Dang khoi dong Web Application...
"C:\ProgramData\anaconda3\envs\envt2\python.exe" run_webapp.py
pause
