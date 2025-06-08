#!/bin/bash

set -e

echo "🔄 Updating package list..."
sudo apt update

echo "📦 Installing default JDK (Java Development Kit)..."
sudo apt install -y default-jdk

echo "✅ Java installation complete. Verifying versions:"
java -version
javac -version

echo "🔍 Detecting JAVA_HOME path..."
JAVA_PATH=$(readlink -f /usr/bin/java | sed "s:/bin/java::")

echo "📁 Backing up current /etc/environment..."
sudo cp /etc/environment /etc/environment.backup

echo "📝 Writing JAVA_HOME to /etc/environment..."
# Remove any existing JAVA_HOME
sudo sed -i '/^JAVA_HOME=/d' /etc/environment
# Append new JAVA_HOME
echo "JAVA_HOME=\"$JAVA_PATH\"" | sudo tee -a /etc/environment

echo "♻️ You must log out and back in (or reboot) for JAVA_HOME to take effect."
echo "✅ Java setup completed successfully!"