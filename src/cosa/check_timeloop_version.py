import subprocess
import logging
import os
import cosa.utils as utils
logger = utils.logger


def check_timeloop_version():
    supported_timeloop_commit="d2e83e9"
    try:
        _TIMELOOP_DIR = os.path.expanduser(os.environ['TIMELOOP_DIR'])
        command = "git rev-parse --short HEAD"
        process = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE)
        output, error = process.communicate()

    except KeyError:
        logger.warning(f"""TIMELOOP_DIR environment variable unspecified for Timeloop version checking.
    Please set TIMELOOP_DIR environment variable to the path of your Timeloop repo using:
        export TIMELOOP_DIR=<path_to_timeloop>
    Current CoSA version is only compatible with Timeloop commit {supported_timeloop_commit}.
    """)

    timeloop_commit = output.decode().strip()
    if timeloop_commit != supported_timeloop_commit:
        logger.warning(f"""Timeloop version mismatched!
    Current Timeloop version: {timeloop_commit}. Expected Timeloop version: {supported_timeloop_commit}.""")
if __name__ == "__main__":
    check_timeloop_version()
