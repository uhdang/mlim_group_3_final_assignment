from db.db import DataBase
from data_preprocessing.category import Category
import os

# Load Data
file_path = os.getcwd() + "/data/"
dbase = DataBase(file_path)

# baskets data
baskets_data = dbase.load_basket_data()

# Product Category Table
catClass = Category(baskets_data)
prods_cat_table = catClass.generate_product_category_table()

# Split Data
baskets_train, baskets_test, coupons_train, coupons_test = dbase.generate_split_data_with_category(prods_cat_table)

print(baskets_train.head(2))
print(baskets_test.head(2))
print(coupons_train.head(2))
print(coupons_test.head(2))
