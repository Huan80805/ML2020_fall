# Package
    1. scikit-learn == 0.22.2.post1
    2. xgboost == 0.90
    3. mlxtend == 0.14.0

# How to run

## Testing
a. run

    bash test.sh <train.csv>  <test.csv> <output.csv>

<train.csv> : the training dataset.
<test.csv> : the testing dataset.
<output.csv> : the output csv file name.


## Training
a. run

    bash train.sh  <train.csv>  <test.csv> 

<train.csv> : the training dataset.
<test.csv> : the testing dataset.

For example:

        bash test.sh train.csv  test.csv output.csv
b. description
it produces the models (.sav files) inside the /src.


# Src
1. python files for submodels:
a. files

        pred_house_pricing_8.py 
        pred_house_pricing_9.py 
        pred_house_pricing_12.py 
        pred_house_pricing_13.py 
        pred_house_pricing_14.py 

b. run:

    python3 <submodel_python_file>  <train.csv>  <test.csv>  <output_csv_file> <model_dir>
    
<submodel_python_file> : the python file name listed in the a. files ;
<train.csv> : the training dataset.
<test.csv> : the testing dataset.
<output_csv_file> : the output csv file name.
<model_dir> :the directory path of the model it uses.

for example:

    python3 src/pred_house_pricing_8.py train.csv test.csv src/submission_8.csv model/House_Pricing_8_mdl/

2. python file for blending
a. file

        house_pricing_blending_1.py
        
        
b. run

    python3 house_pricing_blending_1.py <submodel_output_dir> <output_csv_file>
    
<submodel_output_dir>: the directory where (submission_8.csv, submission_9.csv, submission_12.csv, submission_13.csv, submission_14.csv)  are at.

<output_csv_file> : the output csv file name.

c. description:
It reads the submodel's outputs (submission_8.csv, submission_9.csv, submission_12.csv, submission_13.csv, submission_14.csv) and blend together.
