import pandas as pd
import pickle
import os
import numpy as np
from dataloader import Dataloader
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
    

def generate_coupons():
    """
    Setting up dynamic tables for predictions in week 90
    """
    
    
    path = os.getcwd() + "/../data/"
    # create dataloader object which loads data which creates baskets and coupons
    data = Dataloader(path)
    # Create Categories for products
    data.create_category_table(10000)
    
    shopper_list = list(range(2000))
    shopper_chunks = [shopper_list[i:i + 100] for i in range(0, len(shopper_list), 100)]

    X_train_list = list()
    y_train_list = list()

    # generates training date over weeks 0,...89
    for idx, shopper in enumerate(shopper_chunks):
        print(f"shopper_chunk index: {idx}")
        X_train, y_train = data.baskets_train_prediction(shopper)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        del X_train, y_train

    X_train_df = pd.concat(X_train_list, ignore_index=True)
    X_train_df['shopper'] = X_train_df['shopper'].astype('category')
    X_train_df['product'] = X_train_df['product'].astype('category')
    y_train_df = pd.concat(y_train_list, ignore_index=True)
    y_train_df = y_train_df.astype('category')
    
    # training data with tuned hyperparameters
    with open(os.getcwd() + "/pickle/best_param.pickle", 'rb') as f:
            best_params = pickle.load(f)
    lgb_clf = LGBMClassifier(**best_params)
    lgb_clf.fit(X_train.drop('week'), y_train)
    
        
    # create table for prediction   
    pred_table = pd.DataFrame(
        {
            'week': np.full(250, 90),
            'shopper': np.full(250, shopper),
            'product': list(range(250))
        }
    )
    
    n_coupons = 5
    shoppers=list(range(2000))
    discounts = [0.15, 0.2, 0.25, 0.3]
    # Randomly initialize (product, discount)
    prod_discount_dict = {
        0: (0, 0.15),
        1: (1, 0.2),
        2: (2, 0.2),
        3: (3, 0.25),
        4: (4, 0.3)
    }
    # save all coupons for each shopper
    shopper_coupon_dict = {shopper: {} for shopper in shoppers}
    expected_revenue = 0
    
    
    for shopper in shoppers:
        prod_discount_dict = {
            0: (0, 0.15),
            1: (1, 0.2),
            2: (2, 0.2),
            3: (3, 0.25),
            4: (4, 0.3)
        }   
        for coupon in range(n_coupons):
            for product in list(range(250)):
                for discount in discounts:
                    prod_discount_dict_temp = prod_discount_dict.copy()
                    prod_discount_dict_temp[coupon] = (product, discount)
                    X_predict = create_prediction_tables(shopper, prod_discount_dict_temp)
                    prob = lgb_clf.predict_proba(X_predict)[:,1]
                    expected_revenue_temp = np.matmul(X_predict['price'], prob)
                    if expected_revenue_temp > expected_revenue:
                        expected_revenue = expected_revenue_temp
                        prod_discount_dict = prod_discount_dict_temp
        shopper_coupon_dict[shopper] = prod_discount_dict
        
    return shopper_coupon_dict
                    
    
def create_prediction_tables(shopper, prod_discount_dict):
    """
    Creates table with same features as training table for one shopper and products=0,...,249.
    The prod_discount_dict sets the discount for the given product in the dictionary
    """
    pass
    