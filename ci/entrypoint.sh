#!/bin/bash
# Read in the file of environment settings
source /venv/bin/activate
# Then run the CMD
exec "$@"
