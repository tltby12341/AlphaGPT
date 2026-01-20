#!/bin/bash

# AlphaGPT Server Setup Script
# Works on Ubuntu/Debian

echo "=========================================="
echo "   AlphaGPT Server Setup (Tencent Cloud)  "
echo "=========================================="

# 1. Check for sudo
if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
elif [ "$EUID" -ne 0 ]; then
    echo "Error: This script requires root privileges or sudo."
    exit 1
else
    SUDO=""
fi

# 2. Detect Package Manager and Update System
echo "[1/5] Detecting OS and updating system packages..."

if command -v apt-get >/dev/null 2>&1; then
    PKG_MGR="apt-get"
    $SUDO $PKG_MGR update
    $SUDO $PKG_MGR install -y python3 python3-venv python3-pip git htop
elif command -v dnf >/dev/null 2>&1; then
    PKG_MGR="dnf"
    $SUDO $PKG_MGR check-update
    $SUDO $PKG_MGR install -y python3 python3-pip git htop
elif command -v yum >/dev/null 2>&1; then
    PKG_MGR="yum"
    $SUDO $PKG_MGR check-update
    $SUDO $PKG_MGR install -y python3 python3-pip git htop
else
    echo "Error: Unsupported package manager. Please install dependencies manually."
    exit 1
fi

echo "Using package manager: $PKG_MGR"

# 3. Check Python Version (Must be >= 3.10)
echo "[2/5] Verifying Python version..."
PY_VER=$(python3 -c"import sys; print(sys.version_info.major, sys.version_info.minor)")
SYS_PY_MAJOR=$(echo $PY_VER | cut -d' ' -f1)
SYS_PY_MINOR=$(echo $PY_VER | cut -d' ' -f2)

if [ "$SYS_PY_MAJOR" -lt 3 ] || ([ "$SYS_PY_MAJOR" -eq 3 ] && [ "$SYS_PY_MINOR" -lt 10 ]); then
    echo "Error: Python 3.10+ is required. Current version: Python $SYS_PY_MAJOR.$SYS_PY_MINOR"
    echo "Please upgrade your OS (Ubuntu 22.04+ recommended) or install Python 3.10 manually."
    exit 1
fi

# 4. Create Virtual Environment
echo "[3/5] Creating virtual environment (.venv)..."
# Clear old venv if broken
if [ -d ".venv" ]; then
    if [ ! -f ".venv/bin/activate" ]; then
        echo "Removing broken .venv..."
        rm -rf .venv
    fi
fi

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create venv. Ensure python3-venv is installed."
        exit 1
    fi
    echo "Created .venv"
fi

# 5. Install Dependencies
echo "[4/5] Installing dependencies from requirements.txt..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Error: .venv/bin/activate not found! Setup failed."
    exit 1
fi

# 6. Initialize Database
echo "[5/5] Checking database..."
if [ ! -f "stock_quant.db" ]; then
    echo "Initializing database..."
    python -m data_pipeline.run_pipeline
else
    echo "Database exists. Skipping initialization."
fi

echo "=========================================="
echo "   Setup Complete!                        "
echo "=========================================="
echo "To run the dashboard:"
echo "  source .venv/bin/activate"
echo "  streamlit run dashboard/app.py --server.port 8501"
echo "=========================================="
