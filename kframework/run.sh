#! /bin/bash

set -eux -o pipefail

pushd lesson02
kompile lesson-02-a.k
krun banana.color
krun -cPGM='colorOf(Banana())'
popd

pushd MN-Core2
kompile MN-Core2.k
krun -cPGM='colorOf(Banana())' --depth 0
krun -cPGM='colorOf(Banana())' --depth 1
popd
