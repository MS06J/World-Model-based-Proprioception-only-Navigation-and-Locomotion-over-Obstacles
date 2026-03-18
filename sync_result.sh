#!/bin/bash

# Sync logs directory to remote server
# Remote: <user>@<host>

REMOTE_USER="<your_remote_user>"
REMOTE_HOST="<your_remote_host>"
REMOTE_PATH="<your_remote_path>/WMP/logs"
LOCAL_LOGS="logs"

echo "Syncing ${LOCAL_LOGS} from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.ipynb_checkpoints' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" "${LOCAL_LOGS}"

echo "Sync complete!"
