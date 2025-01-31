import subprocess
import sys
import time


def run_bot():
    while True:
        exit_code = subprocess.call([sys.executable, "bot.py"])
        print(f"Bot exited with code {exit_code}. Restarting in 5s...")
        time.sleep(5)


if __name__ == "__main__":
    run_bot()
