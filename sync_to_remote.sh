#!/bin/bash

# Sync WMP directory to remote server
# Remote: <user>@<host>

REMOTE_USER="<your_remote_user>"
REMOTE_HOST="<your_remote_host>"
REMOTE_PATH="<your_remote_path>/WMP"
LOCAL_PATH="<your_local_path>/WMP"

echo "Syncing ${LOCAL_PATH} to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude '.idea' \
    --exclude '.vscode' \
    --exclude '*.egg-info' \
    --exclude 'logs' \
    --exclude 'wandb' \
    --exclude '.ipynb_checkpoints' \
    --exclude 'build' \
    --exclude 'dist' \
    --exclude '*.mkv' \
    "${LOCAL_PATH}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

echo "Sync complete!"
