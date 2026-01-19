#!/usr/bin/env fish

echo "START: $(date)"
echo "HOST: $(hostname)"
echo "USER: $(whoami)"
echo "PWD: $(pwd)"
echo "COMMAND: $argv"
echo "--------------------------------------------------------------------------"
echo "ENV:"
env
echo "--------------------------------------------------------------------------"

# the first arugment is saved into run
set project $argv[1]
set run $argv[2]
set seed $argv[3]

# exit if environment variable PROJECT_ROOT is not set
if not set -q PROJECT_ROOT
    echo "ERROR: PROJECT_ROOT is not set" >&2
    exit 1
end

cd $PROJECT_ROOT

python ./scripts/train.py \
    project_name=$project \
    network_name=$run \
    seed=$seed

# exit if the last command failed
if test $status -ne 0
    echo "ERROR: Training script failed" >&2
    exit 1
end

python ./scripts/export.py \
    project_name=$project \
    network_name=$run \
    samples_per_event=1024

echo "--------------------------------------------------------------------------"

echo "END: $(date)"
