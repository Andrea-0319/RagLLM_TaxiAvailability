#!/usr/bin/env python3
"""
Launch script for the Telegram bot.
Run: python start_bot.py
"""
import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_tool.telegram_bot import main

if __name__ == "__main__":
    main()