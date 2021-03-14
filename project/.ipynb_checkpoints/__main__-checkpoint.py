from db.Db import DataBase
from data_preprocessing.Category import Category
import os
from tables.Traintable import Traintable
# Load Data

file_path = os.getcwd() + "/data/"
dbase = DataBase(file_path)

# orignal_price
original_price = dbase.original_price()

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

# Train Table
train_t = Traintable(baskets_train, coupons_train, original_price)
X_train, y_train = train_t.split_to_X_train_and_y_train(prods_cat_table)
print(X_train.head(2))
print(y_train.head(2))
