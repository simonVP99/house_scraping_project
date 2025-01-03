#!/bin/bash

# Ensure all necessary environment variables are set
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="/Users/simon"
export SHELL="/bin/bash"

# URL to test for internet connection
TEST_URL="http://www.google.com"

# Path to your virtual environment activation script
VENV_PATH="/Users/simon/Desktop/personal/immo_datacollection/house_scraping_project/.venv/bin/activate"

# Command to run your Python script
PYTHON_COMMAND="python3 /Users/simon/Desktop/personal/immo_datacollection/house_scraping_project/src/main.py 100"

# Log file paths
LOG_FILE="/Users/simon/Desktop/personal/immo_datacollection/house_scraping_project/src/run_script.log"
ENV_VARS_LOG="/Users/simon/Desktop/personal/immo_datacollection/house_scraping_project/src/env_vars.log"

# Log environment variables for debugging
env > $ENV_VARS_LOG

# Start of the script execution
echo "$(date): Script started." >> $LOG_FILE

# Check for internet connection
if curl --output /dev/null --silent --head --fail "$TEST_URL"; then
  echo "$(date): Internet connection is present. Running the Python script." >> $LOG_FILE
  # Activate the virtual environment
  echo "$(date): Activating virtual environment." >> $LOG_FILE
  source $VENV_PATH
  # Run the Python command
  echo "$(date): Running the Python command." >> $LOG_FILE
  $PYTHON_COMMAND >> $LOG_FILE 2>&1
else
  echo "$(date): No internet connection. Exiting." >> $LOG_FILE
fi

# End of the script execution
echo "$(date): Script ended." >> $LOG_FILE
