import subprocess
import logging
import os


def check_timeloop_version():
    try:
        _TIMELOOP_DIR = os.path.expanduser(os.environ['TIMELOOP_DIR'])
        command = "git rev-parse --short HEAD"
        process = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE)
        output, error = process.communicate()

    except KeyError:
        raise Exception("""TIMELOOP_DIR environment variable unspecified for Timeloop version checking.
    Please set TIMELOOP_DIR environment variable to the path of your Timeloop repo using:
        export TIMELOOP_DIR=<path_to_timeloop>
    Current CoSA version is only compatible with Timeloop commit 11920be.
    """)

    timeloop_commit = output.decode().strip()
    if timeloop_commit != "11920be":
        raise Exception(f"""Timeloop version mismatched!
    Current Timeloop version: {timeloop_commit}. Expected Timeloop version: 11920be.""")

if __name__ == "__main__":
    check_timeloop_version()
