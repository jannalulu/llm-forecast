#!/usr/local/bin bash

export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

[ -f venv/bin/activate ] || python3 -m virtualenv venv
source venv/bin/activate
