#!/bin/sh
pypy3 hyfd.py -m -el 0.001 data/diagnostics.csv
pypy3 hyfd.py -m -el 0.001 data/forestfires.csv
pypy3 hyfd.py -m -el 0.001 data/abalone.csv
pypy3 hyfd.py -m -el 0.001 data/pglw00.original.csv
pypy3 hyfd.py -m -el 0.001 data/servo.original.csv
pypy3 hyfd.py -m -el 0.001 data/adult.data.csv
pypy3 hyfd.py -m -el 0.001 data/caulkins.csv
pypy3 hyfd.py -m -el 0.001 data/hughes.original.csv
pypy3 hyfd.py -m -el 0.001 data/cmc.data.csv
pypy3 hyfd.py -m -el 0.001 data/credit.data.csv
pypy3 hyfd.py -m -el 0.001 data/ncvoter_1001r_19c.csv
pypy3 hyfd.py -m -el 0.001 data/hepatitis.csv
pypy3 hyfd.py -m -el 0.001 data/horse.csv -s ';'
pypy3 hyfd.py -m -el 0.001 data/mushroom.csv
pypy3 hyfd.py -m -el 0.001 data/flight_1k.csv -s ';'
pypy3 hyfd.py -m -el 0.001 data/fd-reduced-30.csv