#!/usr/bin/env bash

: ${1:?Please supply a file} # ${1} is the zip being fed to the script -- a little user-friendliness never went amiss
DIR="${1%.zip}" # this is where the current zip will be unzipped
mkdir -p "${DIR}" || exit 1 # might fail if it's a file already
unzip -n -d "${DIR}" "${1}" # unzip current zip to target directory
find "${DIR}" -type f -name '*.zip' -print0 | xargs -0 -n1 "${0}" # scan the target directory for more zips and recursively call this script (via ${0})
rm -r ${DIR}/*/**.zip
