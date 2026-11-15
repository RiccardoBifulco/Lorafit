"""Write the demo checkpoints to a directory for manual inspection."""

import argparse
from pathlib import Path

from lora_fitcheck._demo import build_demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        for path in build_demo(args.directory):
            print(path)
    except OSError as exc:
        parser.exit(2, f"could not write demo checkpoints: {exc}\n")
