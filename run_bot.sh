#!/bin/bash
# Crypto Insight Bot — auto-start wrapper
# Managed by launchd: com.crypto.insightbot

cd "$(dirname "$0")"

# Use python3 from the shell environment
export PATH="/usr/bin:/usr/local/bin:/Users/phamlongvu/Library/Python/3.9/bin:$PATH"
export PYTHONPATH="$(pwd)"
export PYTHONUNBUFFERED=1

exec /usr/bin/python3 telegram_bot.py
