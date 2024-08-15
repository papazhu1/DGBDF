# DGBDF
The source code for paper, “A Dual Granular Balanced Deep Forest Model for Effective Drug Combination Prediction”.

## Requirements
The requirements for DGBDF can be found in the requirements.txt file.
```bash
Python==3.9
```

# Usage

## Dataset Preparation
Unzip the dataset.zip file to obtain the required dataset.

## Training the DGBDF
You may run the demo.py in model folder to train the DGBDF model by 5-fold stratified cross-validation.

```bash
cd model
python demo.py
```

## Get results

The mean performance of DGBDF can be available once the 5-fold training is completed.