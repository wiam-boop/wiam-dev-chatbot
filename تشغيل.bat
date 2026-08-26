@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist venv (
    echo جارٍ إنشاء البيئة الافتراضية لأول مرة...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo جارٍ تثبيت المكتبات المطلوبة...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo ==============================================
echo   افتح المتصفح على: http://localhost:5000
echo   لإيقاف السيرفر: اضغط CTRL+C
echo ==============================================
echo.

python app.py

pause
