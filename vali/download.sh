#!/usr/bin/env bash
# Download the Vali country data the shipped datasets were built from.
#
# Countries were chosen for spread of coverage age, not for map aesthetics --
# old un-refreshed imagery is the only source of the old copyright years:
#   old gen2/gen3   MN KG LK BD GH SN UG KE NG BO PE CL RO BG UA AL
#   earliest SV     US AU JP NZ IT ES FR MX
#   modern/gen4     IN TW ZA PH TH ID
#
# Roughly 25 GB in total. Set the download folder first:
#   vali set-download-folder ~/vali-data
#
# `script` supplies a pseudo-terminal: vali polls the console for a keypress and
# crashes outright when stdin is not a terminal.
set -uo pipefail

export PATH="$PATH:$HOME/.dotnet/tools"
DATA_DIR="${VALI_DOWNLOAD_FOLDER:-$HOME/vali-data}/Vali"

COUNTRIES="${*:-MN KG LK BD GH SN UG KE NG BO PE CL RO BG UA AL US AU JP NZ IT ES FR MX IN TW ZA PH TH ID LU}"

for c in $COUNTRIES; do
  printf '=== %s  %s ===\n' "$c" "$(date +%H:%M:%S)"
  script -qec "vali download --country $c" /dev/null 2>&1 | grep -E "finished|%" | tail -2
  [ -d "$DATA_DIR" ] && du -sh "$DATA_DIR" | cut -f1
done
printf 'done  %s\n' "$(date +%H:%M:%S)"
