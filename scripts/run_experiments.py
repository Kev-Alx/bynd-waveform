"""Standalone entry point: experiments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beyondwaveform.cli import main

if __name__ == "__main__":
    main(["experiments", *sys.argv[1:]])

