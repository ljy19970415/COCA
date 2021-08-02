# COCA
Here is the source code of COCA system.

## COCA.py
COCA.py is the entry of the COCA system.
You run the code by command: python COCA.py
You can enter parameters in commandline, and there are three parameters that is used in our article, which are "dataset", "net", and "p".
dataset can be either 'CUB' or 'Stanford Dogs'
net can be 'resnet50', 'vgg16' or 'mobilenet'
p is the accuracy of amateurs, 1.0 means no noise, 0.9 means accuracy is 90%

## annotate.py
annotate.py includes the annotation process of amateurs and experts.
It has the firstBatch funtion which is the first annotation batch process and restBatch function for the rest annotation batches.

## cluster.py
cluster.py includes the feature extraction, clustering process at the beginning of the annotation process. And it also generates confidence, updates the metric and produces amateur and expert rankings in the rest batches.

## lmnn.py
lmnn.py implements the LMNN metric learning models.

## **_records/&&
**_records/&& store the records during the annotation process. ** represents the dataset and && represents the feature extraction. 
records.txt and records.csv will record annotation status per annotation batch, including:
batch_number, precision, cost, expert-annotated number, amateur-annotated number, amateur produced labeled data, total labeled data, the standard deviation of sample amount for each category, amount of dicovered categories.

## **_records/&&/middle_point
If the execution is interrupted in the middle. Set the isContinue parameter to be 1, the code will load the information in middle_point directory and continue the execution from the break point.

## CUB and Stanford Dogs
Thoses directories store the results of the feature extraction and the first clustering. Due to the limitation of file size in github, the feature extraction results can't be uploaded.
You can reach me by misslei@mail.ustc.edu.cn to ask for the feature file.
