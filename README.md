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
   - $ run.sh

## Pipeline

The pipeline includes the following steps:
1. Build data and run hyper-tuning
1. Train and make a prediction

