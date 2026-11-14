#!/bin/bash

# Resolve the project root from this script's own location so the script works
# regardless of where the repo is checked out or which user runs it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# URL to test for internet connection
TEST_URL="http://www.google.com"

# Path to your virtual environment activation script
VENV_PATH="$PROJECT_DIR/.venv/bin/activate"

# Command to run your Python script
PYTHON_COMMAND="python3 -u $SCRIPT_DIR/main.py 100"

# Log file paths
LOG_FILE="$SCRIPT_DIR/run_script.log"
ENV_VARS_LOG="$SCRIPT_DIR/env_vars.log"

# Log environment variables for debugging
env > "$ENV_VARS_LOG"

# Rotate the log once it passes 20 MB, keeping one gzipped generation
if [ -f "$LOG_FILE" ] && [ "$(stat -f%z "$LOG_FILE")" -gt 20971520 ]; then
  gzip -c "$LOG_FILE" > "${LOG_FILE}.1.gz"
  : > "$LOG_FILE"
fi

# Start of the script execution
echo "$(date): Script started." >> "$LOG_FILE"

# Check for internet connection
if curl --output /dev/null --silent --head --fail "$TEST_URL"; then
  echo "$(date): Internet connection is present. Running the Python script." >> "$LOG_FILE"
  # Activate the virtual environment
  echo "$(date): Activating virtual environment." >> "$LOG_FILE"
  source "$VENV_PATH"
  # Run the Python command
  echo "$(date): Running the Python command." >> "$LOG_FILE"
  $PYTHON_COMMAND >> "$LOG_FILE" 2>&1
else
  echo "$(date): No internet connection. Exiting." >> "$LOG_FILE"
fi

# End of the script execution
echo "$(date): Script ended." >> "$LOG_FILE"
