#!/bin/bash

cp $1 src/train.csv

cp $2 src/test.csv

cd ./src

python3 house_pricing_8.py
python3 house_pricing_9.py
python3 house_pricing_12.py
python3 house_pricing_13.py
python3 house_pricing_14.py

cd ../
