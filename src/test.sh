#!/bin/bash
# Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
# All rights reserved. This file is part of LBM-Bench.
# Use of this source code is governed by a MIT style
# license that can be found in the LICENSE file.
#
# Verification harness for LBM-Bench. Adapted from the upstream
# lbm-benchmark-kernels test.sh / test-verification.sh.
#
# Builds the binary with -DVERIFICATION enabled for both double (dp) and
# single (sp) precision, then runs every kernel reported by `-l` against
# the built-in Poiseuille channel-flow check (analytical solution,
# L2 error norm, tolerance 0.1).
#
# Limitations vs. upstream:
#   - no -pin; use OMP_PLACES / OMP_PROC_BIND / taskset for pinning
#   - blocked kernels use their compile-time block size (no -blk sweep)
set -u

Toolchain="${1:-CLANG}"

case "$Toolchain" in
  -h|--help|-help)
    echo "Usage: $0 [TOOLCHAIN]"
    echo "  TOOLCHAIN: GCC | CLANG | ICX (default: CLANG)"
    exit 0
    ;;
esac

ScriptDir="$(cd "$(dirname "$0")" && pwd)"
RepoDir="$(cd "$ScriptDir/.." && pwd)"
cd "$RepoDir"

# Bootstrap config.mk if missing (first make invocation creates it then errors).
[ -f config.mk ] || make TOOLCHAIN="$Toolchain" >/dev/null 2>&1 || true
if [ ! -f config.mk ]; then
  echo "ERROR: failed to bootstrap config.mk."
  exit 1
fi

# Enable -DVERIFICATION in config.mk; restore on exit.
ConfigBackup="$(mktemp -t lbmbench-config.mk.XXXXXX)"
cp config.mk "$ConfigBackup"
Tmp="$(mktemp -t lbmbench-test.XXXXXX)"
cleanup() {
  cp "$ConfigBackup" config.mk
  rm -f "$ConfigBackup" "$Tmp"
}
trap cleanup EXIT

sed -i 's|^[[:space:]]*#[[:space:]]*OPTIONS[[:space:]]*+=[[:space:]]*-DVERIFICATION|OPTIONS += -DVERIFICATION|' config.mk
if ! grep -q '^OPTIONS *+= *-DVERIFICATION' config.mk; then
  echo "OPTIONS += -DVERIFICATION" >>config.mk
fi

TestsTotal=0
TestsFailed=0
TestsSucceeded=0
Binary=""

run_kernel() {
  local K="$1"
  TestsTotal=$((TestsTotal + 1))
  echo -n "$Binary -V -k $K ... "
  if "$Binary" -V -k "$K" >"$Tmp" 2>&1; then
    echo "OK"
    TestsSucceeded=$((TestsSucceeded + 1))
  else
    local rc=$?
    echo "FAILED (exit=$rc)"
    cat "$Tmp"
    TestsFailed=$((TestsFailed + 1))
  fi
}

run_precision() {
  local P="$1"
  Binary="./lbmbench-$Toolchain-$P"

  echo "#"
  echo "# [test.sh] building $Binary with -DVERIFICATION (PRECISION=$P)"
  echo "#"

  if ! make TOOLCHAIN="$Toolchain" PRECISION="$P" clean >/dev/null 2>&1; then :; fi
  if ! make TOOLCHAIN="$Toolchain" PRECISION="$P"; then
    echo "ERROR: build failed for PRECISION=$P."
    exit 1
  fi
  if [ ! -x "$Binary" ]; then
    echo "ERROR: expected binary $Binary not found."
    exit 1
  fi

  local Kernels
  Kernels="$("$Binary" -l | awk '/^Available kernels:/{flag=1; next} flag{print $1}')"
  if [ -z "$Kernels" ]; then
    echo "ERROR: could not parse kernel list from $Binary -l."
    exit 1
  fi

  for K in $Kernels; do
    run_kernel "$K"
  done
}

run_precision dp
run_precision sp

echo "#"
echo "# Tests total: $TestsTotal  succeeded: $TestsSucceeded  failed: $TestsFailed"
echo "#"

[ "$TestsFailed" -eq 0 ]
