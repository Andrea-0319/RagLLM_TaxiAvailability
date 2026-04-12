#!/usr/bin/env python3
"""
Launch script for the Streamlit app.
Run: python start_streamlit.py
"""
import sys
import os
import subprocess

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app_path = os.path.join(os.path.dirname(__file__), "llm_tool", "StreamlitRania", "app.py")

if __name__ == "__main__":
    subprocess.run(["streamlit", "run", app_path])