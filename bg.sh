#!/bin/sh
# usage: bg.sh <logfile> <cmd...>  — start a detached job in ~/mfp
cd ~/mfp || exit 1
log=$1; shift
setsid nohup nice -n 10 "$@" > "$log" 2>&1 < /dev/null &
echo started $!
