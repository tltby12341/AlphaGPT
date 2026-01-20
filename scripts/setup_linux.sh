#!/bin/bash

# AlphaGPT Server Setup Script
# Works on Ubuntu/Debian

echo "=========================================="
echo "   AlphaGPT Server Setup (Tencent Cloud)  "
echo "=========================================="

# 1. Update System
echo "[1/5] Updating system packages..."
sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv python3-pip git htop

# 2. Check Python Version
echo "[2/5] Verifying Python version..."
python3 --version

# 3. Create Virtual Environment
echo "[3/5] Creating virtual environment (.venv)..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
else
    python3 -m venv .venv
    echo "Created .venv"
fi

# 4. Install Dependencies
echo "[4/5] Installing dependencies from requirements.txt..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Initialize Database
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
