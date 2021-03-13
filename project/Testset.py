import pandas as pd
import numpy as np


class Testset:

    def __init__(self, baskets_test, coupons_test, num_shoppers=100):
        self.baskets_test = baskets_test
        self.coupons_test = coupons_test
        self.num_shoppers = num_shoppers

    def generate_full_df_test(self):
        full_df_test = (self.baskets_test
                        .merge(
            self.coupons_test,
            on=["week", "shopper", "product", "category"],
            how="outer",
            indicator=True)
                        .sort_values(by=["week", "shopper", "product"])
                        .reset_index(drop=True)
                        .replace(["left_only", "right_only", "both"], ["No", "Yes", "Yes"])
                        .rename(columns={"_merge": "coupon"})
                        )

        full_df_test["discount"].fillna(0, inplace=True)
        return full_df_test

    def generate_featureless_testing_set(self, prods_cat_table):
        full_df_test = self.generate_full_df_test()

        df1 = pd.DataFrame({'key': np.ones(len(self.baskets_test["week"].unique())), 'week': self.baskets_test["week"].unique()})
        df2 = pd.DataFrame({'key': np.ones(self.num_shoppers), 'shopper': list(range(self.num_shoppers))})
        df3 = pd.DataFrame({'key': np.ones(250), 'product': list(range(250))})

        featureless_testing_set = (pd
            .merge(df1, df2, on='key')
            .merge(df3, on='key')
            .merge(prods_cat_table, on='product')
            .merge(full_df_test, on=['week', 'shopper', 'product', 'category'], how='left')[full_df_test.columns]
            )

        return featureless_testing_set

    def generate_training_set(self, featureless_testing_set,
                              original_price,
                              total_count_of_product,
                              reordered_product,
                              category_count,
                              reordered_category,
                              coupon_in_same_category,
                              average_price_per_shopper,
                              average_basket_size,
                              unique_products_per_shopper,
                              unique_categories_per_shopper,
                              training_set
        # ratio_of_reordered_products_per_shopper,
        # ratio_of_reordered_categories_per_shopper,
                              ):
        testing_set = (featureless_testing_set
                        .merge(original_price, on=['product'], how='left')
                        .merge(total_count_of_product, on=['shopper', 'product'], how='left')
                        .merge(reordered_product, on=['shopper', 'product'], how='left')
                        .merge(category_count, on=['shopper', 'category'], how='left')
                        .merge(reordered_category, on=['shopper', 'category'], how='left')
                        .merge(coupon_in_same_category, on=['week', 'shopper', 'category'], how='left')
                        .merge(average_price_per_shopper, on=['shopper'], how='left')
                        .merge(average_basket_size, on=['shopper'], how='left')
                        .merge(unique_products_per_shopper, on=['shopper'], how='left')
                        .merge(unique_categories_per_shopper, on=['shopper'], how='left')
                       # .merge(ratio_of_reordered_products_per_shopper(), on=['shopper'], how='left')
                       # .merge(ratio_of_reordered_categories_per_shopper(), on=['shopper'], how='left')
                        )

        testing_set['discount'].fillna(0, inplace=True)
        testing_set['price'].fillna(testing_set.original_price * (1 - testing_set.discount / 100), inplace=True)
        testing_set['reordered_product'].fillna(0, inplace=True)
        testing_set['reordered_category'].fillna(0, inplace=True)
        testing_set['coupon'].fillna('No', inplace=True)
        testing_set['target'].fillna(0, inplace=True)
        testing_set['coupon_in_same_category'].fillna('No', inplace=True)

        # Take weeks_since_prior_order from last available week in training and add 1
        last_week_since_prior_product_order = training_set.groupby(
            ['shopper', 'product']).weeks_since_prior_product_order.last() + 1
        last_week_since_prior_category_order = training_set.groupby(
            ['shopper', 'category']).weeks_since_prior_category_order.last() + 1

        testing_set = (testing_set
                  .merge(last_week_since_prior_product_order, on=['shopper', 'product'])
                  .merge(last_week_since_prior_category_order, on=['shopper', 'category'])
                  )

        return testing_set

    def split_testingset_to_X_test_and_y_test(self, testing_set):
        y_test = testing_set.pop("target")
        X_test = testing_set.drop("week", inplace=True, axis=1)

        # categorical = X_test.select_dtypes(exclude=np.number).columns.tolist()
        # for cats in categorical:
        #     X_test[cats] = X_test[cats].astype("category")
        return X_test, y_test
