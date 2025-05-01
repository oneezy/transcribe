@echo off
title 🔄 Resetting Poetry Env (CUDA-ready)

echo 🧼 Checking if current shell is inside a venv...
where python | findstr /I ".venv" > nul
if %errorlevel%==0 (
    echo 🔌 Detected active virtualenv — deactivating...
    call deactivate
) else (
    echo ✅ No active virtualenv
)

echo 🔪 Deleting old .venv and __pycache__...
rmdir /s /q .venv
rmdir /s /q __pycache__

echo ⚙️  Setting Poetry to use in-project virtualenv...
poetry config virtualenvs.in-project true

echo 🚀 Installing from pyproject.toml...
poetry install

echo 🧠 Verifying GPU + CUDA...
poetry run python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| CUDA version:', torch.version.cuda)"

echo 🟢 Done. Press any key to exit.
pause
