#!/bin/bash
# Deploy: copy updated files from Desktop source → crypto_bot, then restart bot
SRC="/Users/phamlongvu/Desktop/dư đoán xu hướng giá (trend price prediction)/files"
DST="/Users/phamlongvu/crypto_bot"

echo "📦 Deploying from Desktop → crypto_bot..."
cp "$SRC"/*.py "$DST/"
cp "$SRC"/.env "$DST/"
cp "$SRC"/requirements.txt "$DST/"

echo "🔄 Restarting bot..."
launchctl unload ~/Library/LaunchAgents/com.crypto.insightbot.plist 2>/dev/null
sleep 2
launchctl load ~/Library/LaunchAgents/com.crypto.insightbot.plist
sleep 3

if launchctl list | grep -q "com.crypto.insightbot"; then
    echo "✅ Bot restarted successfully"
    launchctl list | grep crypto
else
    echo "❌ Failed to restart"
fi
