@echo off
title Resetting Poetry Env (CUDA-ready)
chcp 65001 > nul

:: 📁 Ensure .vscode directory exists
if not exist ".vscode" (
    mkdir .vscode
)

:: ⚙️ Write settings.json for VS Code workspace
echo {> .vscode\settings.json
echo.  "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe",>> .vscode\settings.json
echo.  "python.terminal.activateEnvironment": true,>> .vscode\settings.json
echo.  "python.analysis.extraPaths": ["${workspaceFolder}"]>> .vscode\settings.json
echo }>> .vscode\settings.json

echo 🧼 Checking if current shell is inside a venv...
where python | findstr /I ".venv" > nul
if %errorlevel%==0 (
    echo 🔌 Detected active virtualenv — deactivating...
    call deactivate
) else (
    echo ✅ No active virtualenv
)

echo 🔪 Deleting old .venv and __pycache__...
rmdir /s /q .venv > nul 2>&1
rmdir /s /q __pycache__ > nul 2>&1

:: 📦 Check if Poetry is installed
where poetry > nul 2>&1
if %errorlevel% neq 0 (
    echo 🚧 Poetry not found. Installing it now...
    curl -sSL https://install.python-poetry.org | python -
    setx PATH "%APPDATA%\Python\Scripts;%PATH%"
    echo 🆗 Poetry installed. Restart this terminal or run install.bat again.
    pause
    exit /b
)

echo 🚀 Installing from pyproject.toml...
poetry install

echo 🧠 Verifying GPU + CUDA...
poetry run python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| CUDA version:', torch.version.cuda)"

echo 🟢 Done. Press any key to exit.
pause
