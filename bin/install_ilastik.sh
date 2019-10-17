#!/usr/bin/env bash
set -e

HERE=$(readlink -f "$0")
PROJECT_ROOT=$(dirname $(dirname ${HERE}))
REQUIREMENTS_PROD="${PROJECT_ROOT}/requirements/prod.txt"
REQUIREMENTS_DEV="${PROJECT_ROOT}/requirements/test.txt"

read -p "Are you sure? May harm your current python environment. ([y]/n): " -n 1 -r
if [[  $REPLY =~ ^[Yy]$ ]]
then
    conda install ilastik-dependencies-no-solvers -c ilastik-forge -c conda-forge -y
    conda install nifty-with-gurobi -c ilastik-forge -c conda-forge -y
    pip install -r ${REQUIREMENTS_PROD}
fi
