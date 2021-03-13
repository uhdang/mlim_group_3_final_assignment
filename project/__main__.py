from Db import DataBase
from Category import Category
import os
from Feature import Feature
from Trainset import Trainset
from Testset import Testset
from Trainer import Trainer

file_path = os.getcwd() + "/data/"
dbase = DataBase(file_path)

baskets_data, coupons_data = dbase.load_basket_coupon_data()

# orignal_price
original_price = dbase.original_price(baskets_data)

# Product Category Table
catClass = Category(baskets_data)
prods_cat_table = catClass.generate_product_category_table()

baskets_train, baskets_test, coupons_train, coupons_test = dbase.split_data(baskets_data, coupons_data)

baskets_train, baskets_test, coupons_train, coupons_test = dbase.generate_split_data_with_category(prods_cat_table, baskets_train, baskets_test, coupons_train, coupons_test)

print(baskets_train.head(2))
print(baskets_test.head(2))
print(coupons_train.head(2))
print(coupons_test.head(2))


## individual features
print("feature engineering begin")
feat = Feature(baskets_train, coupons_train)
total_count_of_product = feat.total_count_of_product()
reordered_product = feat.reordered_product()
category_count = feat.category_count()
reordered_category = feat.reordered_category()
coupon_in_same_category = feat.coupon_in_same_category()
average_price_per_shopper = feat.average_price_per_shopper()
average_basket_size = feat.average_basket_size()
unique_products_per_shopper = feat.unique_products_per_shopper()
unique_categories_per_shopper = feat.unique_categories_per_shopper()
# ratio_of_reordered_products_per_shopper = feat.ratio_of_reordered_products_per_shopper()
# ratio_of_reordered_categories_per_shopper = feat.ratio_of_reordered_categories_per_shopper()
print("feature engineering finished")

# Train Table
train_t = Trainset(baskets_train, coupons_train, original_price)

## featureless_training_set
featureless_training_set = train_t.generate_featureless_training_set(prods_cat_table)
print("----- featureless training set -----")
print(featureless_training_set.head(2))

## Generate training set
training_set = train_t.populate_features(
    featureless_training_set,
    original_price,
    total_count_of_product,
    reordered_product,
    category_count,
    reordered_category,
    coupon_in_same_category,
    average_price_per_shopper,
    average_basket_size,
    unique_products_per_shopper,
    unique_categories_per_shopper
#     ratio_of_reordered_products_per_shopper,
#     ratio_of_reordered_categories_per_shopper,
)
print("===== training_set =====")
print(training_set.head(2))


# X_train, y_train
X_train, y_train = train_t.split_trainingset_to_X_train_and_y_train(training_set)
print(X_train.head(2))
print(y_train.head(2))


# # testing table
# test_t = Testset(baskets_test, coupons_test)
#
# # full_df_test
# full_df_test = test_t.generate_full_df_test()
#
# # Generate testing set
# testing_set = test_t.generate_training_set(
#     prods_cat_table,
#     original_price,
#     total_count_of_product,
#     reordered_product,
#     category_count,
#     reordered_category,
#     coupon_in_same_category,
#     average_price_per_shopper,
#     average_basket_size,
#     unique_products_per_shopper,
#     unique_categories_per_shopper,
#     training_set
# #     ratio_of_reordered_products_per_shopper,
# #     ratio_of_reordered_categories_per_shopper,
# )
#
# X_test, y_test = test_t.split_testingset_to_X_test_and_y_test(testing_set)
# print(X_test.head(2))
# print(y_test.head(2))
#
#
# # Trainer
#
# trainer = Trainer(X_train, y_train, X_test, y_test)
# model = trainer.fit_model()
