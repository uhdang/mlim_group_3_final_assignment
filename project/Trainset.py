import pandas as pd
import numpy as np


class Trainset:

    def __init__(self, baskets_train, coupons_train, num_weeks=88, num_shoppers=100):
        self.baskets_train = baskets_train
        self.coupons_train = coupons_train
        self.num_weeks = num_weeks
        self.num_shoppers = num_shoppers

    def generate_full_df_train(self):
        full_df_train = (self.baskets_train
                         .merge(
            self.coupons_train,
            on=["week", "shopper", "product", "category"],
            how="outer",
            indicator=True)
                         .sort_values(by=["week", "shopper", "product"])
                         .reset_index(drop=True)
                         .replace(["left_only", "right_only", "both"], ["No", "Yes", "Yes"])
                         .rename(columns={"_merge": "coupon"})
                         )
        full_df_train["discount"].fillna(0, inplace=True)
        return full_df_train

    def generate_featureless_training_set(self, prods_cat_table):
        full_df_train = self.generate_full_df_train()

        df1 = pd.DataFrame({'key': np.ones(self.num_weeks), 'week': list(range(self.num_weeks))})
        df2 = pd.DataFrame({'key': np.ones(self.num_shoppers), 'shopper': list(range(self.num_shoppers))})
        df3 = pd.DataFrame({'key': np.ones(250), 'product': list(range(250))})

        featureless_training_set = (pd
            .merge(df1, df2, on='key')
            .merge(df3, on='key')
            .merge(prods_cat_table, on='product')
            .merge(full_df_train, on=['week', 'shopper', 'product', 'category'], how='left')[full_df_train.columns]
            )
        return featureless_training_set


    def populate_features(self, featureless_training_set,
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
                              # ratio_of_reordered_products_per_shopper,
                              # ratio_of_reordered_categories_per_shopper
                              ):

        training_set = (featureless_training_set
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

        training_set['discount'].fillna(0, inplace=True)
        training_set['price'].fillna(training_set.original_price * (1 - training_set.discount / 100), inplace=True)
        training_set['reordered_product'].fillna(0, inplace=True)
        training_set['reordered_category'].fillna(0, inplace=True)
        training_set['coupon'].fillna('No', inplace=True)
        training_set['target'].fillna(0, inplace=True)
        training_set['coupon_in_same_category'].fillna('No', inplace=True)

        # Count weeks since last order of that product
        addkey = training_set.groupby(['shopper', 'product'])["target"].apply(
            lambda x: x.eq(1).shift().fillna(0).cumsum())
        training_set['weeks_since_prior_product_order'] = training_set["target"].eq(0).groupby(
            [training_set['shopper'], training_set['product'], addkey]).cumcount().add(1)  # .cumsum()

        # Count weeks since last order of that category
        addkey = training_set.groupby(['shopper', 'category'])["target"].apply(
            lambda x: x.eq(1).shift().fillna(0).cumsum())
        training_set['weeks_since_prior_category_order'] = training_set["target"].eq(0).groupby(
            [training_set['shopper'], training_set['category'], addkey]).cumcount().add(1)  # .cumsum()

        return training_set

    def split_trainingset_to_X_train_and_y_train(self, training_set):
        y_train = training_set.pop("target")
        X_train = training_set.drop("week", inplace=True, axis=1)

        # categorical = X_train.select_dtypes(exclude=np.number).columns.tolist()
        # for cats in categorical:
        #     X_train[cats] = X_train[cats].astype("category")
        #
        return X_train, y_train
