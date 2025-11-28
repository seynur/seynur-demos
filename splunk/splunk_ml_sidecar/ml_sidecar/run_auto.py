#!/usr/bin/env python3
"""
------------------------------------------------------------------------------
 File: run_auto.py
 Project: Splunk ML Sidecar
 Description:
     Entry point for running the full ML Sidecar automated pipeline.

     This script acts as the main executable that:
       • Loads pipeline configuration
       • Performs ingestion → training → inference → scoring → export
       • Writes results to Splunk KVStore when enabled
       • Is intended to be called via CLI, cron, or Splunk scripted execution

 Usage:
     $ python run_auto.py

 Notes:
     - All heavy logic is inside the pipeline modules (etc/pipeline/*)
     - This file intentionally stays minimal to act as a clean entrypoint
-------------------------------------------------------------------------------
"""

from core.pipeline import run_auto_pipeline


def main():
    """
    Execute the ML Sidecar automated pipeline.

    This function wraps the orchestration call to allow easier
    future extension (argument parsing, flags, modes, dry-run, etc.).
    """
    run_auto_pipeline()


if __name__ == "__main__":
    main()