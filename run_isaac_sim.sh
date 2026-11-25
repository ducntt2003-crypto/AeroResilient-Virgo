#!/usr/bin/env bash
# ==============================================================================
# VIRGO UAV: LAUNCH ISAAC SIM 3D INTERACTIVE FLIGHT
# ==============================================================================
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$DIR/Virgo_RL" ]; then
    cd "$DIR/Virgo_RL"
elif [ -d "$DIR/../Virgo_RL" ]; then
    cd "$DIR/../Virgo_RL"
fi

PYTHON_EXEC="/home/duc/miniconda3/envs/env_isaaclab/bin/python"

echo "================================================================================"
echo " 🚀 LAUNCHING VIRGO UAV 3D INTERACTIVE SIMULATION IN ISAAC SIM"
echo " Working Directory: $(pwd)"
echo " Python Binary:     $PYTHON_EXEC"
echo "================================================================================"

"$PYTHON_EXEC" scripts/interactive_fly_virgo.py "$@"
