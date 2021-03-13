import pandas as pd

class DataBase:
    cat_columns = ["shopper", "product"]
    split_week = 88

    def __init__(self, file_path):
        self.file_path = file_path

    def load_basket_data(self, with_target=True):
        baskets = pd.read_parquet(self.file_path + "baskets.parquet")
        if with_target:
            baskets["target"] = 1
        return baskets

    def load_coupon_data(self):
        return pd.read_parquet(self.file_path + "coupons.parquet")

    def load_basket_coupon_data(self):
        baskets = self.load_basket_data()
        coupons = self.load_coupon_data()
        return baskets, coupons

    def split_data(self, baskets, coupons, num_shoppers=100):

        baskets_train = baskets.loc[(baskets["shopper"].isin(list(range(num_shoppers)))) & (baskets["week"] <= self.split_week), :]
        baskets_test = baskets.loc[(baskets["shopper"].isin(list(range(num_shoppers)))) & (baskets["week"] > self.split_week), :]

        coupons_train = coupons.loc[(coupons["shopper"].isin(list(range(num_shoppers)))) & (coupons["week"] <= self.split_week), :]
        coupons_test = coupons.loc[(coupons["shopper"].isin(list(range(num_shoppers)))) & (coupons["week"] > self.split_week), :]

        baskets_train[self.cat_columns] = baskets_train[self.cat_columns].astype('category')
        baskets_test[self.cat_columns] = baskets_test[self.cat_columns].astype('category')
        coupons_train[self.cat_columns] = coupons_train[self.cat_columns].astype('category')
        coupons_test[self.cat_columns] = coupons_test[self.cat_columns].astype('category')

        return baskets_train, baskets_test, coupons_train, coupons_test

    def generate_split_data_with_category(self, prods_cat_table, baskets_train, baskets_test, coupons_train, coupons_test):

        baskets_train = baskets_train.merge(prods_cat_table, on=['product'], how='left')
        baskets_test = baskets_test.merge(prods_cat_table, on=['product'], how='left')

        coupons_train = coupons_train.merge(prods_cat_table, on=['product'], how='left')
        coupons_test = coupons_test.merge(prods_cat_table, on=['product'], how='left')

        return baskets_train, baskets_test, coupons_train, coupons_test

    def original_price(self, baskets):
        return baskets.groupby('product', as_index=False)["price"].max().rename(
            columns={'price': 'original_price'})


    # def load_all_data(self):
    #     baskets = self.load_basket_data()
    #     coupons = self.load_coupon_data()
    #     coupon_index = pd.read_parquet(self.file_path + "coupon_index.parquet")
    #     return baskets, coupons, coupon_index


        
        