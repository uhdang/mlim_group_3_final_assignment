import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


class Dataloader:

    def __init__(self,
                 path
    ):
        self.baskets = self._load_data(path, 'baskets.parquet')
        self.coupons = self._load_data(path, 'coupons.parquet')
        self.weeks = None
        self.shoppers = None
        self.baskets_train = None
        self.baskets_test = None
        self.coupons_train = None
        self.coupons_test = None
        self.feature_dict = None
        self.prods_cat_table = None
        self.num_cat = 25

    def train_test_split(self, weeks, shoppers):
        self.weeks = weeks
        self.shoppers = shoppers
        self.baskets_train = self._split(self.baskets, with_target=True)
        self.baskets_test = self._split(self.baskets, is_test=True, with_target=True)

        self.coupons_train = self._split(self.coupons)
        self.coupons_test = self._split(self.coupons, is_test=True)

        # return self.baskets_train, self.baskets_test, self.coupons_train, self.coupons_test


    def add_categories(self):
        if self.prods_cat_table is None: self.create_category_table()

        self.baskets_train = self.baskets_train.merge(self.prods_cat_table, on=['product'], how='left')
        self.baskets_test = self.baskets_test.merge(self.prods_cat_table, on=['product'], how='left')

        self.coupons_train = self.coupons_train.merge(self.prods_cat_table, on=['product'], how='left')
        self.coupons_test = self.coupons_test.merge(self.prods_cat_table, on=['product'], how='left')

        return self.baskets_train, self.baskets_test, self.coupons_train, self.coupons_test


    def create_category_table(self, n_shoppers=1000):
        # Step 1: create a list of baskets
        shoppers_p2v = list(range(n_shoppers))
        baskets_p2v = self.baskets.loc[
            self.baskets['shopper'].isin(shoppers_p2v)
            ][['week', 'shopper', 'product']]

        basket_values = baskets_p2v.sort_values(['week', 'shopper']).values
        keys = baskets_p2v['week'].astype(str) + '_' + baskets_p2v['shopper'].astype(str)
        _, index = np.unique(keys, True)
        basket_arr = np.split(basket_values[:, 2].astype('str'), np.sort(index))[1:]

        # Step 2
        datastr = DataStreamer(basket_arr)
        model = Word2Vec(
            sentences=datastr,
            size=100,
            window=20, # max. size of basket
            min_count=1,
            negative=2,
            sample=0,
            workers=4,
            sg=1
        )
        product_keys = [str(product) for product in range(250)]
        product_vectors = model.wv[product_keys]

        # Step 3: Generating product categories
        kmeans = KMeans(n_clusters=25, random_state=0).fit(product_vectors)
        products = [product for product in range(250)]
        prods_cat_table = pd.DataFrame(data=products, columns=["product"])
        prods_cat_table["category"] = kmeans.labels_
        prods_cat_table[['product', 'category']] = prods_cat_table[['product', 'category']].astype('category')

        self.prods_cat_table = prods_cat_table


    def make_featured_data(self):
        X_train = self._make_full_table(self.baskets_train, self.coupons_train)
        X_test = self._make_full_table(self.baskets_test, self.coupons_test)

        X_train = self._merge_features(X_train, self.feature_dict)
        X_test = self._merge_features(X_test, self.feature_dict)

        # Count weeks since last order of that product
        addkey = X_train.groupby(['shopper','product'])['target'].apply(lambda x : x.eq(1).shift().fillna(0).cumsum())
        X_train['weeks_since_prior_product_order'] = X_train['target'].eq(0).groupby([X_train['shopper'], X_train['product'], addkey]).cumcount().add(1)

        # Count weeks since last order of that category
        cat_target = X_train.groupby(['week', 'shopper', 'category'])['target'].max().to_frame().reset_index()
        addkey = cat_target.groupby(['shopper', 'category']).target.apply(lambda x : x.eq(1).shift().fillna(0).cumsum())
        cat_target['weeks_since_prior_category_order'] = cat_target.target.eq(0).groupby([cat_target['shopper'], cat_target['category'], addkey]).cumcount().add(1)
        X_train = X_train.merge(cat_target[['week', 'shopper', 'category', 'weeks_since_prior_category_order']], on=['week', 'shopper', 'category'], how='left')

        # Take weeks_since_prior_order from last available week in training and add 1
        last_week_since_prior_product_order = X_train.groupby(['shopper', 'product'])['weeks_since_prior_product_order'].last() + 1
        last_week_since_prior_category_order = X_train.groupby(['shopper', 'category'])['weeks_since_prior_category_order'].last() + 1

        X_test = (X_test
                .merge(last_week_since_prior_product_order, on=['shopper', 'product'])
                .merge(last_week_since_prior_category_order, on=['shopper', 'category'])
        )

        X_train.drop('week', inplace=True, axis=1)
        X_test.drop('week', inplace=True, axis=1)
        y_train = X_train.pop('target')
        y_test = X_test.pop('target')

        return X_train, y_train, X_test, y_test


    def _load_data(self, path, name):
        data = pd.read_parquet(path + name)
        data = self._categorize_data(data, remove_cat=False)

        return data


    def _categorize_data(self, data, remove_cat=True):
        cat_columns = ['shopper', 'product']
        for column in cat_columns:
            data[column] = data[column].astype('category')
            if remove_cat:
                data[column] = data[column].cat.remove_unused_categories()

        return data


    def _original_price(self):
        original_price = (self.baskets
                          .groupby('product', as_index=False)['price']
                          .max()
                          .rename(columns={'price': 'original_price'})
                        )

        return original_price


    def _split(self, data, is_test=False, with_target=False):
        if is_test:
            data_split = data.loc[
                (data['shopper'].isin(self.shoppers))
                & (data['week']==self.weeks)
            ]
        else:
            data_split = data.loc[
                (data['shopper'].isin(self.shoppers))
                & (data['week']<self.weeks)
            ]

        data_split = self._categorize_data(data_split)

        if with_target:
            data_split["target"] = 1

        return data_split


    def _make_full_table(self, basket, coupon):

        combined_df = self._combine_basket_coupon(basket, coupon)
        weeks = list(basket["week"].unique())

        df1 = pd.DataFrame({
            'key':np.ones(len(weeks)),
            'week':weeks
        })
        df2 = pd.DataFrame({
            'key':np.ones(len(self.shoppers)),
            'shopper':self.shoppers
        })
        df3 = pd.DataFrame({
            'key':np.ones(250), 'product':list(range(250))
        })

        full_df = (pd
                   .merge(df1, df2, on='key')
                   .merge(df3, on='key')
                   .merge(self.prods_cat_table, on='product')
                   .merge(combined_df,
                          on=['week', 'shopper', 'product', 'category'],
                          how='left'
                   )[combined_df.columns]
        )

        return full_df


    def _merge_features(self, full_df, feature):
        original_price = self._original_price()

        full_df = (full_df
                   .merge(original_price, on=['product'], how='left')
                   .merge(feature['total_count_of_product'], on=['shopper', 'product'], how='left')
                   .merge(feature['reordered_product'], on=['shopper', 'product'], how='left')
                   .merge(feature['category_count'], on=['shopper', 'category'], how='left')
                   .merge(feature['reordered_category'], on=['shopper', 'category'], how='left')
                   .merge(feature['coupon_in_same_category'], on=['week', 'shopper', 'category'], how='left')
                   .merge(feature['avg_categorical_discount'], on=['category'], how='left')
                   .merge(feature['num_coupons_week_shopper_category'], on=["week", "shopper"], how="left")
                   .merge(feature['ratio_of_reordered_products_per_shopper'], on=['shopper'], how='left')
                   .merge(feature['ratio_of_reordered_categories_per_shopper'], on=['shopper'], how='left')
                   .merge(feature['average_price_per_shopper'], on=['shopper'], how='left')
                   .merge(feature['average_basket_size'], on=['shopper'], how='left')
                   .merge(feature['unique_products_per_shopper'], on=['shopper'], how='left')
                   .merge(feature['unique_categories_per_shopper'], on=['shopper'], how='left')
        )
        full_df['discount'].fillna(0, inplace=True)
        full_df['price'].fillna(full_df['original_price']*(1-full_df['discount']/100), inplace=True)
        full_df['reordered_product'].fillna(0, inplace=True)
        full_df['reordered_category'].fillna(0, inplace=True)
        full_df['coupon'].fillna('No', inplace=True)
        full_df['target'].fillna(0, inplace=True)
        full_df['coupon_in_same_category'].fillna('No', inplace=True)

        return full_df


    def _combine_basket_coupon(self, basket, coupon):
        combined_df = (basket
                       .merge(
                           coupon,
                           on=['week', 'shopper', 'product', 'category'],
                           how='outer',
                           indicator=True
                       )
                       .sort_values(by=['week', 'shopper', 'product'])
                       .reset_index(drop=True)
                       .replace(['left_only', 'right_only', 'both'], ['No', 'Yes', 'Yes'])
                       .rename(columns={'_merge': 'coupon'})
        )
        combined_df['discount'].fillna(0, inplace=True)

        return combined_df

    def get_avg_categorical_discount(self, original_price):
        c_w_op = self.coupons_train.merge(original_price, on="product", how="left")
        c_w_op["abs_discount_v"] = c_w_op["original_price"] * (c_w_op["discount"] / 100)
        avg_categorical_discount = c_w_op.groupby(["category"], as_index=False)["abs_discount_v"].mean() \
            .rename(columns={"abs_discount_v": "avg_categorical_discount"})
        return avg_categorical_discount

    def get_num_coupons_week_shopper_category(self):
        coupons_week_shopper_category_grouped = self.coupons_train.groupby(["week", "shopper", "category"], as_index=False)[
            "discount"].count()
        coupons_week_shopper_category_grouped["discount"] = coupons_week_shopper_category_grouped["discount"].fillna(0)
        week_shopper_only = coupons_week_shopper_category_grouped[["week", "shopper"]].drop_duplicates(
            subset=["week", "shopper"], keep="last").reset_index(drop=True)
        d_only = coupons_week_shopper_category_grouped["discount"].values
        sliced_by_cat = [d_only[self.num_cat * i:self.num_cat * i + self.num_cat] for i in range(0, len(d_only) // self.num_cat)]
        sliced_df = pd.DataFrame(data=sliced_by_cat, columns=[list(range(self.num_cat))])
        num_coupons_week_shopper_category = pd.concat([week_shopper_only, sliced_df], axis=1)
        return num_coupons_week_shopper_category

    def create_feature_dict(self):
        original_price = self._original_price()

        # Product related features
        total_count_of_product = (self.baskets_train
                                  .groupby(['shopper', 'product'])['product']
                                  .count().to_frame('total_count_of_product')
                                  .reset_index()
        )
        reordered_product = ((self.baskets_train
                              .groupby(['shopper'])['product']
                              .value_counts()>1)
                              .astype(int)
        )

        # Category related features
        category_count = (self.baskets_train
                          .groupby(['shopper', 'category'])['category']
                          .count()
                          .to_frame('category_count')
                          .reset_index()
        )
        reordered_category = ((self.baskets_train
                               .groupby(['shopper'])['category']
                               .value_counts()>1)
                               .astype(int)
        )
        coupon_in_same_category = (self.coupons_train
                                   .loc[:, ['week', 'shopper', 'category']]
                                   .drop_duplicates()
        )
        coupon_in_same_category['coupon_in_same_category'] = 'Yes'

        ## avg_categorical_discount
        avg_categorical_discount = self.get_avg_categorical_discount(original_price)

        ## num_coupons_week_shopper_category
        num_coupons_week_shopper_category = self.get_num_coupons_week_shopper_category()

        # Shopper related features
        average_price_per_shopper = (self.baskets_train
                                     .groupby(['shopper'])['price']
                                     .mean()
                                     .to_frame('average_price_per_shopper')
                                     .reset_index()
        )
        average_basket_size = (self.baskets_train
                               .groupby(['shopper', 'week'])['product']
                               .count().groupby('shopper')
                               .mean()
                               .to_frame('average_basket_size')
                               .reset_index()
        )
        unique_products_per_shopper = (self.baskets_train
                                       .groupby(['shopper'])['product']
                                       .nunique()
        )
        unique_categories_per_shopper = (self.baskets_train
                                         .groupby(['shopper'])['category']
                                         .nunique()
        )
        ratio_of_reordered_products_per_shopper = ((reordered_product.groupby('shopper').sum()
                                                    / unique_products_per_shopper)
                                                .to_frame('ratio_of_reordered_products')
                                                .reset_index()
        )
        ratio_of_reordered_categories_per_shopper = ((reordered_category.groupby('shopper').sum()
                                                    / unique_categories_per_shopper)
                                                    .to_frame('ratio_of_reordered_categories')
                                                    .reset_index()
        )

        # Frame unframed arrays
        reordered_product = reordered_product.to_frame('reordered_product').reset_index()
        reordered_category = reordered_category.to_frame('reordered_category').reset_index()
        unique_products_per_shopper = unique_products_per_shopper.to_frame('unique_products_per_shopper').reset_index()
        unique_categories_per_shopper = unique_categories_per_shopper.to_frame('unique_categories_per_shopper').reset_index()

        self.feature_dict = {
            'total_count_of_product': total_count_of_product,
            'reordered_product': reordered_product,
            'category_count': category_count,
            'reordered_category': reordered_category,
            'coupon_in_same_category': coupon_in_same_category,
            'avg_categorical_discount': avg_categorical_discount,
            'num_coupons_week_shopper_category': num_coupons_week_shopper_category,
            'average_price_per_shopper': average_price_per_shopper,
            'average_basket_size': average_basket_size,
            'unique_products_per_shopper': unique_products_per_shopper,
            'unique_categories_per_shopper': unique_categories_per_shopper,
            'ratio_of_reordered_products_per_shopper': ratio_of_reordered_products_per_shopper,
            'ratio_of_reordered_categories_per_shopper': ratio_of_reordered_categories_per_shopper
        }


class DataStreamer():
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for basket in self.data:
            yield basket.tolist()
