import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from db.Datastreamer import DataStreamer

class Category:

    split_week = 88

    def __init__(self, baskets_data):
        self.baskets_data = baskets_data

    def generate_basket_list(self, num_shoppers=1000):
        baskets_p2v = self.baskets_data.loc[(self.baskets_data["shopper"].isin(list(range(num_shoppers)))) & (self.baskets_data["week"] <= self.split_week), :][["week", "shopper", "product"]]
        basket_values = baskets_p2v.sort_values(["week", "shopper"]).values
        keys = baskets_p2v["week"].astype(str) + "_" + baskets_p2v["shopper"].astype(str)
        _, index = np.unique(keys, True)
        basket_arr = np.split(basket_values[:, 2].astype("str"), np.sort(index))[1:]
        return basket_arr

    def generate_word2vec_model(self, basket_arr=None, size=100, window=20, min_count=1, negative=2, sample=0, sg=1, workers=4):
        if basket_arr == None:
            basket_arr = self.generate_basket_list()

        datastr = DataStreamer(basket_arr)

        model = Word2Vec(
            sentences=datastr,
            size=size,
            window=window,
            min_count=min_count,
            negative=negative,
            sample=sample,
            sg=sg,
            workers=workers
        )

        return model

    def generate_product_category_table(self, w2v_model=None, num_category=25):
        if w2v_model == None:
            w2v_model = self.generate_word2vec_model()

        product_keys = [str(product) for product in range(250)]
        product_vectors = w2v_model.wv[product_keys]

        # Reduce dimention of vectors
        # X_embedded = TSNE(n_components=2).fit_transform(product_vectors)
        kmeans = KMeans(n_clusters=num_category, random_state=0).fit(product_vectors)

        products = [product for product in range(250)]
        prods_cat_table = pd.DataFrame(data=products, columns=["product"])
        prods_cat_table["category"] = kmeans.labels_
        prods_cat_table[['product', 'category']] = prods_cat_table[['product', 'category']].astype('category')
        return prods_cat_table



