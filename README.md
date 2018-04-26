# Classifying Mice Brain Scans
This repo contains the analysis, report, source code and scripts for the final project. Built using Apache Maven, Apache Spark + MLlib, Jupyter Notebooks and python.

## Project Information
* Report: /report
* Raw Train/Test Data: https://drive.google.com/drive/u/1/folders/1EJBgJFmp-FQf2czw9LGImoOhEO2OvOoo
* Our processed Data: https://drive.google.com/drive/u/1/folders/1v6--KV2CVjHcMLXXl4Q_9MXyeTgPna-T (pre-proc-data.zip)

### Project Structure

    ${PROJ_ROOT}
    ├── data # create for runs
    │   └── pre-proc-data #unzipped data from google drive
    ├── src
    │   └── main
    │       └── resources # python preprocessing
    │       └── scala # models, feature extraction, training/testing
    ├── input
    │   ├── L6_1_965381.csv
    │   ├── 10true10false.txt
    │   ├── onlyones.txt
    │   ├── onlyzeros.txt
    │   └── test.zip
    ├── notebooks # notebooks for data exploration, feature extraction, & processing
    │
    └── README.md

Files:
1. convertCsvToImg.py: given a path to oa data file (i.e. ./convertCsvToImg.py L6_1_965381.csv),the output is in text PGM (an image format). The output path is hardcoded in the file.
2. src/main/scala - Apache Spark / Scala (model training and prediction)
3. notebooks/ - Jupyter Notebooks
4. img-output - image output from notebooks visualization
5. data - small input data for testing and preprocessing data zip

### Result comparison
To count the difference between 2 files:

    diff -U 0 file1 file2 | grep ^@ | wc -l

To get the labels of the image files:

    cat data/pre-proc-data/validation/L6_2_982271.csv | awk -F, '{print $NF}' | awk -F. '{print $1}' > img2true

### Subsets of data

Find the rows of your file that has label as one:

    cat L6_1_965381.csv | grep '\.*1$' > onlyones.txt


Find the rows of your file that has label as zero:

    cat L6_1_965381.csv | grep '\.*0$' > onlyzeros.txt
    
Generate a file with 100 "foreground" images and 100 "background" images
    
    cat onlyones.txt > 100true100false.txt
    cat onlyzeros.txt >> 100true100false.txt # This appends to file
    
Now you can use the `100true100false.txt` file in your scripts


## Running Locally
* Open the project in Eclipse IDE (file, import, existing maven project)
* Download our processed data (pre-proc-data) from the drive and put under ml-brains/data in order to run
* In the Makefile, set the following for your env:

        spark.root
        hadoop.root
        class.name - change to model we are running (i.e. Final Classification)

* To run different parameter settings for the model, edit the following:

        max_depth
        max_bins
        num_trees
        aws.num.workers

* Run commands:

        jar
        make alone-t # t for training, c for classification/prediction
        
## Running on EMR
* Create bucket on S3 with subfolders ls-pdp/proj (the project folder, see aws.proj.dir in Makefile)
* Add the local data/pre-proc-data locally to EMR bucket
* Put one of the raw data files (6GB) given to us on the google drive locally under root/data/raw-data
* In the Makefile, set the following for your env (and EMR):

        aws.bucket.name
        aws.subnet.id

* Run the following commands:

        make
        make upload-app-aws # for jar
        make upload-all-aws # for conf, raw and preprocess data
        make cloud-t # t for training

* To run classification/prediction:

        make cloud-c

* Download output files with the following command:

        make download-output-aws


### S3 Data Structure (bucketname/ls-pdp/proj)

    ├── models
    ├── pre-proc-data
    │   ├── test
    │   │   └── L6_2_982271.csv
    │   ├── train
    │   │   ├── L6_1_965381.csv
    │   │   ├── L6_3_978153.csv
    │   │   ├── L6_4_978153.csv
    │   │   └── L6_6_972760.csv
    │   └── validation
    │       └── L6_5_new.csv
    └── raw-data
    └── L6_1_965381.csv


