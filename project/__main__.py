from db.Db import DataBase
from data_preprocessing.Category import Category
import os
# Load Data
from datasets.Trainset import Trainset
from feature_engineering.Feature import Feature
from datasets.Testset import Testset
from Trainer import Trainer

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


## individual features
print("feature engineering begin")
feat = Feature(baskets_train, coupons_train)

total_count_of_product = feat.total_count_of_product()
print(total_count_of_product.head(2))

reordered_product = feat.reordered_product()
print(reordered_product.head(2))

category_count = feat.category_count()
print(category_count.head(2))

reordered_category = feat.reordered_category()
print(reordered_category.head(2))

coupon_in_same_category = feat.coupon_in_same_category()
print(coupon_in_same_category.head(2))

ratio_of_reordered_products_per_shopper = feat.ratio_of_reordered_products_per_shopper()

ratio_of_reordered_categories_per_shopper = feat.ratio_of_reordered_categories_per_shopper()

average_price_per_shopper = feat.average_price_per_shopper()

average_basket_size = feat.average_basket_size()

unique_products_per_shopper = feat.unique_products_per_shopper()

unique_categories_per_shopper = feat.unique_categories_per_shopper()

print("feature engineering finished")

# Train Table
train_t = Trainset(baskets_train, coupons_train, original_price)

## full_df_train
full_df_train = train_t.generate_full_df_train()
print("----- full_df_train -----")
print(full_df_train.head(2))

## Generate training set
training_set = train_t.generate_training_set(
    prods_cat_table,
    original_price,
    total_count_of_product,
    reordered_product,
    category_count,
    reordered_category,
    coupon_in_same_category,
    ratio_of_reordered_products_per_shopper,
    ratio_of_reordered_categories_per_shopper,
    average_price_per_shopper,
    average_basket_size,
    unique_products_per_shopper,
    unique_categories_per_shopper,
)
print("===== training_set =====")
print(training_set.head(2))

# X_train, y_train
X_train, y_train = train_t.split_trainingset_to_X_train_and_y_train(training_set)
print(X_train.head(2))
print(y_train.head(2))


# testing table
test_t = Testset(baskets_test, coupons_test)

# full_df_test
full_df_test = test_t.generate_full_df_test()

# Generate testing set
testing_set = test_t.generate_training_set(
    prods_cat_table,
    original_price,
    total_count_of_product,
    reordered_product,
    category_count,
    reordered_category,
    coupon_in_same_category,
    ratio_of_reordered_products_per_shopper,
    ratio_of_reordered_categories_per_shopper,
    average_price_per_shopper,
    average_basket_size,
    unique_products_per_shopper,
    unique_categories_per_shopper,
    training_set
)

X_test, y_test = test_t.split_testingset_to_X_test_and_y_test(testing_set)
print(X_test.head(2))
print(y_test.head(2))


# Trainer

trainer = Trainer(X_train, y_train, X_test, y_test)
model = trainer.fit_model()
