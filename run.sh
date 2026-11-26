#!/usr/bin/env bash
# 사용법: ./run.sh server | ask "질문" | eval 1 | eval 2
set -e
export PYTHONUTF8=1
case "$1" in
  server) uvicorn src.server:app --reload --port 8000 ;;
  ask)    shift; python -m src.agent "$@" ;;
  eval)   python -m evaluation.run_eval --round "${2:-1}" ;;
  *)      echo "usage: ./run.sh server | ask \"질문\" | eval 1|2" ;;
esac
