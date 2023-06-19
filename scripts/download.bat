@echo off
setlocal EnableDelayedExpansion

set PYTHON_PATH=F:\Python39\python.exe
set SCRIPT_PATH=F:\python\pystock\finance_git\01_loadData\download_stock.py
set CONFIG_PATH=F:\python\pystock\finance_git\01_loadData\config.json

%PYTHON_PATH% %SCRIPT_PATH% --config %CONFIG_PATH%

endlocal
