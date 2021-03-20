import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import random


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
        self.prods_vec_table = None
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
        
        self.baskets_train = self.baskets_train.merge(self.prods_cat_table, on=['product'], how='left')#.merge(self.prods_vec_table, on=['product'], how='left')
        self.baskets_test = self.baskets_test.merge(self.prods_cat_table, on=['product'], how='left')#.merge(self.prods_vec_table, on=['product'], how='left')

        self.coupons_train = self.coupons_train.merge(self.prods_cat_table, on=['product'], how='left')
        self.coupons_test = self.coupons_test.merge(self.prods_cat_table, on=['product'], how='left')
        
        # return self.baskets_train, self.baskets_test, self.coupons_train, self.coupons_test
    
    
    def create_category_table(self, n_shoppers=10000):
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
        prods_vec_table = pd.DataFrame({int(product_key): product_vector for (product_key, product_vector) in zip(product_keys, product_vectors)})
        prods_vec_table = prods_vec_table.T.rename_axis('product').reset_index()
        prods_vec_table['product'] = prods_vec_table['product'].astype('category')
        
        # Step 3: Generating product categories
        kmeans = KMeans(n_clusters=25, random_state=0).fit(product_vectors)
        products = [product for product in range(250)]
        prods_cat_table = pd.DataFrame(data=products, columns=["product"])
        prods_cat_table["category"] = kmeans.labels_
        prods_cat_table[['product', 'category']] = prods_cat_table[['product', 'category']].astype('category')

        self.prods_vec_table = prods_vec_table
        self.prods_cat_table = prods_cat_table


    def make_featured_data(self):
        X_train = self._make_full_table(self.baskets_train, self.coupons_train)
        X_test = self._make_full_table(self.baskets_test, self.coupons_test)

        X_train = self._merge_features(X_train, self.feature_dict)
        X_test = self._merge_features(X_test, self.feature_dict)

        ############################## weeks since last order #################################
        # Count weeks since last order of that product
        X_train = self._weeks_since_last_order(X_train, 'product')

        # Count weeks since last order of that category
        cat_target = X_train.groupby(['week', 'shopper', 'category'])['target'].max().to_frame().reset_index()
        cat_target = self._weeks_since_last_order(cat_target, 'category')
        X_train = X_train.merge(cat_target[['week', 'shopper', 'category', 'weeks_since_prior_category_order']], on=['week', 'shopper', 'category'], how='left')

        # Take weeks_since_prior_order from last available week in training and add 1
        last_week_since_prior_product_order = X_train.groupby(['shopper', 'product'])['weeks_since_prior_product_order'].last() + 1
        last_week_since_prior_category_order = X_train.groupby(['shopper', 'category'])['weeks_since_prior_category_order'].last() + 1

        X_test = (X_test
                .merge(last_week_since_prior_product_order, on=['shopper', 'product'])
                .merge(last_week_since_prior_category_order, on=['shopper', 'category'])
        )
        ############################### rolling order count/frequencies #####################################
        windows = [3, 5, 15, 30]
        for window in windows:
            X_train = self._rolling_order_count(X_train, 'product', window, True)
            X_temp = self._rolling_order_count(X_train.copy(), 'product', window, False)
            X_test = X_test.merge(
                X_temp.loc[X_temp['week']==(self.weeks-1), ['shopper', 'product', 'count_of_product_order_last_' + str(window) + '_weeks']], 
                on=['shopper', 'product'], 
                how='left'
            )
            
            cat_target = X_train.groupby(['week', 'shopper', 'category'])['target'].sum().to_frame().reset_index()
            cat_target = self._rolling_order_count(cat_target, 'category', window, True)
            X_train = X_train.merge(cat_target[['week', 'shopper', 'category', 'count_of_category_order_last_' + str(window) + '_weeks']], on=['week', 'shopper', 'category'], how='left')
            X_temp_cat = self._rolling_order_count(cat_target, 'category', window, False)
            X_test = X_test.merge(
                X_temp_cat.loc[X_temp_cat['week']==(self.weeks-1), ['shopper', 'category', 'count_of_category_order_last_' + str(window) + '_weeks']], 
                on=['shopper', 'category'], 
                how='left'
            )
        #####################################################################################################
        
        categorical = ['shopper', 'product', 'category', 'coupon', 'coupon_in_same_category']
        for cats in categorical:
          X_train[cats] = X_train[cats].astype('category')
          X_test[cats] = X_test[cats].astype('category')

        X_train.drop('week', inplace=True, axis=1)
        X_test.drop('week', inplace=True, axis=1)
        y_train = X_train.pop('target')
        y_test = X_test.pop('target')

        return X_train, y_train, X_test, y_test

    
    def _weeks_since_last_order(self, X_train, feature):
        addkey = X_train.groupby(['shopper', feature])['target'].apply(lambda x : x.eq(1).shift().fillna(0).cumsum())
        X_train['weeks_since_prior_' + feature + '_order'] = X_train['target'].eq(0).groupby([X_train['shopper'], X_train[feature], addkey]).cumcount().add(1) 
        return X_train
    
    
    def _rolling_order_count(self, X_train, feature, window, is_train):
        X_train['count_of_' + feature + '_order_last_' + str(window) + '_weeks'] = X_train.groupby(['shopper', feature])['target'].apply(lambda x: x.rolling(window).sum().shift(is_train)).fillna(0)
        return X_train
        

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


    # def _make_full_table(self, basket, coupon):

    #     combined_df = self._combine_basket_coupon(basket, coupon)
    #     weeks = list(basket["week"].unique())

    #     df1 = pd.DataFrame({
    #         'key':np.ones(len(weeks)),
    #         'week':weeks
    #     })
    #     df2 = pd.DataFrame({
    #         'key':np.ones(len(self.shoppers)),
    #         'shopper':self.shoppers
    #     })
    #     df3 = pd.DataFrame({
    #         'key':np.ones(250), 'product':list(range(250))
    #     })

    #     full_df = (pd
    #                .merge(df1, df2, on='key')
    #                .merge(df3, on='key')
    #                .merge(self.prods_cat_table, on='product')
    #                .merge(self.prods_vec_table, on='product')
    #                .merge(combined_df, 
    #                       on=['week', 'shopper', 'product', 'category'], 
    #                       how='left'
    #                )   
    #     )
    #     full_df = full_df.loc[:, full_df.columns!='key']
        
    #     return full_df

    def _make_full_table(self, basket, coupon):
      combined_df = self._combine_basket_coupon(basket, coupon)
      week_list = list(basket["week"].unique())

      weeks = []
      shoppers = []
      products = []

      for week in week_list:
        for shopper in self.shoppers:
          product_list = combined_df.loc[(combined_df['week']==week) & (combined_df['shopper']==shopper), 'product'].to_list()
          n = len(product_list)//2
          weeks = weeks + np.full(n, week).tolist()
          shoppers = shoppers + np.full(n, shopper).tolist()
          products = products + random.sample([i for i in range(250) if i not in product_list], n)

      negative_basket = pd.DataFrame({
        'week': pd.Series(weeks, dtype='int'),
        'shopper': pd.Series(shoppers, dtype='category'),
        'product': pd.Series(products, dtype='category')
      })

      full_df = (negative_basket
                .merge(self.prods_cat_table, on='product')
                .merge(combined_df, on=['week', 'shopper', 'product', 'category'], how='outer')
                .merge(self.prods_vec_table, on='product')    
      )

      return full_df


    def _merge_features(self, full_df, feature):
        original_price = self._original_price()

        full_df = (full_df
                   .merge(original_price, on=['product'], how='left')
                   .merge(feature['ratio_product_count'], on=['shopper', 'product'], how='left')
                   .merge(feature['ratio_category_count'], on=['shopper', 'category'], how='left')
                   .merge(feature['reordered_product'], on=['shopper', 'product'], how='left')
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
        full_df['ratio_product_count'].fillna(0, inplace=True)
        full_df['ratio_category_count'].fillna(0, inplace=True)    # maybe leave NA and to drop NAs for negative sampling
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
        coupons_week_shopper_category_grouped = pd.concat([self.coupons_train, self.coupons_test]).groupby(["week", "shopper", "category"], as_index=False)[
            "discount"].count()
        coupons_week_shopper_category_grouped["discount"] = coupons_week_shopper_category_grouped["discount"].fillna(0)
        week_shopper_only = coupons_week_shopper_category_grouped[["week", "shopper"]].drop_duplicates(
            subset=["week", "shopper"], keep="last").reset_index(drop=True)
        d_only = coupons_week_shopper_category_grouped["discount"].values
        sliced_by_cat = [d_only[self.num_cat * i:self.num_cat * i + self.num_cat] for i in range(0, len(d_only) // self.num_cat)]
        sliced_df = pd.DataFrame(data=sliced_by_cat, columns=[f"c_cat_{cat}" for cat in range(self.num_cat)])
        num_coupons_week_shopper_category = pd.concat([week_shopper_only, sliced_df], axis=1)
        return num_coupons_week_shopper_category

    def create_feature_dict(self):
        original_price = self._original_price()

        # Product related features
        ratio_product_count = (self.baskets_train
                                  .groupby(['shopper', 'product'])['product']
                                  .count()/self.weeks
        ).to_frame('ratio_product_count').reset_index()
                                  
        reordered_product = ((self.baskets_train
                              .groupby(['shopper'])['product']
                              .value_counts()>1)
                              .astype(int)
        )

        # Category related features
        ratio_category_count = (self.baskets_train
                          .groupby(['shopper', 'category'])['category']
                          .count()/self.weeks
        ).to_frame('ratio_category_count').reset_index()
        
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
            'ratio_product_count': ratio_product_count,
            'ratio_category_count': ratio_category_count,
            'reordered_product': reordered_product,
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

def create_combined_dict(X_train_list, y_train_list, X_test_list, y_test_list, cv_dict):
  if X_train_list==list():
    pass
  else:
    X_train_df = pd.concat(X_train_list, ignore_index=True)
    X_train_df['shopper'] = X_train_df['shopper'].astype('category')
    y_train_df = pd.concat(y_train_list, ignore_index=True)
    y_train_df = y_train_df.astype('category')
    cv_dict['X_train'].append(X_train_df)
    cv_dict['y_train'].append(y_train_df)
    
  X_test_df = pd.concat(X_test_list, ignore_index=True)
  X_test_df['product'] = X_test_df['product'].astype('category')
  X_test_df['shopper'] = X_test_df['shopper'].astype('category')
  y_test_df = pd.concat(y_test_list, ignore_index=True)
  y_test_df = y_test_df.astype('category')
  cv_dict['X_test'].append(X_test_df)
  cv_dict['y_test'].append(y_test_df)

  return cv_dict
