#!/bin/bash

python3 src/pred_house_pricing_8.py $1 $2 src/submission_8.csv model/House_Pricing_8_mdl/
python3 src/pred_house_pricing_9.py $1 $2 src/submission_9.csv model/House_Pricing_9_mdl/
python3 src/pred_house_pricing_12.py $1 $2 src/submission_12.csv model/House_Pricing_12_mdl/
python3 src/pred_house_pricing_13.py $1 $2 src/submission_13.csv model/House_Pricing_13_mdl/
python3 src/pred_house_pricing_14.py $1 $2 src/submission_14.csv model/House_Pricing_14_mdl/

python3 src/house_pricing_blending_1.py src/ $3


