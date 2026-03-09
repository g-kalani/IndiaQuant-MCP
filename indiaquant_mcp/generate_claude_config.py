#!/usr/bin/env python3

import os
import sys
import json

def main():
    server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "server.py"))
    python_path = sys.executable

    config = {
        "mcpServers": {
            "indiaquant": {
                "command": python_path,
                "args": [server_path],
                "env": {
                    "NEWSAPI_KEY": "your_newsapi_key_here",
                    "ALPHA_VANTAGE_KEY": "your_alpha_vantage_key_here"
                }
            }
        }
    }

    print("=" * 60)
    print("Paste this into your Claude Desktop config file:")
    print()
    print("Mac:   ~/Library/Application Support/Claude/claude_desktop_config.json")
    print("Win:   %APPDATA%\\Claude\\claude_desktop_config.json")
    print("Linux: ~/.config/Claude/claude_desktop_config.json")
    print()
    print("=" * 60)
    print(json.dumps(config, indent=2))
    print("=" * 60)
    print()
    print(f"Python path detected: {python_path}")
    print(f"Server path:          {server_path}")
    print()
    print("Remember to replace API key placeholders with your real keys!")

if __name__ == "__main__":
    main()
