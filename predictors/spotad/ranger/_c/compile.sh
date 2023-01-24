#!/usr/bin/env bash

# make sure cmake is installed
sudo apt-get install cmake libboost-all-dev -y

# checkout ranger in a tmp location
rm -rf /tmp/ranger
git clone https://github.com/imbs-hl/ranger.git /tmp/ranger

# pull the correct version from the repository
pushd . # remember current location
cp *.patch /tmp/ranger # copy all patch files
cd /tmp/ranger
git checkout 47bdf0b501a5ad8e446dea9b1ecfbb8f64f9b3b6 .
git apply 01_tab_delimiter_support.patch # add patch to support tab delimiters
git apply 02_fix_case_weights.patch # Fix bug for case weights parameter
git apply 03_gz_support.patch # Fix bug for case weights parameter
popd # revert to original location

# build ranger
pushd . # remember current location
rm -rf /tmp/ranger-build
mkdir /tmp/ranger-build
cd /tmp/ranger-build
cmake /tmp/ranger/cpp_version
make
popd # revert to original location

# copy the ranger executable here
cp /tmp/ranger-build/ranger ./gzranger

# clean tmp folders
rm -rf /tmp/ranger
rm -rf /tmp/ranger-build
