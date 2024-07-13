#!/usr/bin/env bash

cd "$(dirname "$(readlink -f "$0")")"
source setup-env.sh

python3 run.py "$@"
