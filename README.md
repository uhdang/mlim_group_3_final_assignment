# MLiM Group 3 Final Assignment

## Requirements

1. `conda`
1. This project is implemented with Python 3.8
1. Necessary data - `baskets.parquet`, `coupon_index.parquet`, `coupons.parquet`. in a folder `data`

## Setup
1. Create and activate virtual environment
   - $ conda create -n hu_mlim_group_3 python=3.8
   - $ conda activate hu_mlim_group_3
2. Install dependencies
   - $ pip install -r requirements.txt
3. Run pipeline
   - $ sh run.sh

## Pipeline

The pipeline includes the following steps:
1. Build data and run hyper-tuning
1. Train and make a prediction

## File Info

- project
    - dataloader.py : dataloader class including data-loading and data-generation logic as well as feature engineering
    - generate_data.py : data generation and cross-validation steps to produce hyperparameter-tuned parameters
    - train_cv.ipynb : Jupyter Notebook used for development
