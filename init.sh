#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

export PYTHONPATH="$DIR:$PYTHONPATH"
export PATH="$DIR:$PATH"
export PATH="$DIR/bin:$PATH"

export PYTHONPATH="$DIR/train:$PYTHONPATH"
export PATH="$DIR/train:$PATH"

export PYTHONPATH="$DIR/utilities:$PYTHONPATH"
export PATH="$DIR/utilities:$PATH"
