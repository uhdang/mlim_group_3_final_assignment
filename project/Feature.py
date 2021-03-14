class Feature:

    def __init__(self, baskets_train, coupons_train):
        self.baskets_train = baskets_train
        self.coupons_train = coupons_train

    # Product related features:
    def total_count_of_product(self):
        total_count_of_product = self.baskets_train.groupby(["shopper", "product"])["product"].count().to_frame(
            "total_count_of_product").reset_index()
        return total_count_of_product

    def reordered_product(self):
        reordered_product = (self.baskets_train.groupby(["shopper"])["product"].value_counts() > 1).astype(int)
        # reordered_product = reordered_product.to_frame("reordered_product").reset_index()
        return reordered_product

    # Category related features:
    def category_count(self):
        category_count = self.baskets_train.groupby(["shopper", "category"])["category"].count().to_frame(
            "category_count").reset_index()
        return category_count

    def reordered_category(self):
        reordered_category = (self.baskets_train.groupby(["shopper"])["category"].value_counts() > 1).astype(int)
#         reordered_category = reordered_category.to_frame("reordered_category").reset_index()
        return reordered_category

    def coupon_in_same_category(self):
        coupon_in_same_category = self.coupons_train.loc[:, ["week", "shopper", "category"]].drop_duplicates()
        coupon_in_same_category["coupon_in_same_category"] = "Yes"
        return coupon_in_same_category

    # Shopper related features:
    def average_price_per_shopper(self):
        average_price_per_shopper = self.baskets_train.groupby(["shopper"])["price"].mean().to_frame(
            "average_price_per_shopper").reset_index()
        return average_price_per_shopper

    def average_basket_size(self):
        average_basket_size = self.baskets_train.groupby(["shopper", "week"])["product"].count().groupby(
            "shopper").mean().to_frame("average_basket_size").reset_index()
        return average_basket_size

    def unique_products_per_shopper(self):
        unique_products_per_shopper = self.baskets_train.groupby(["shopper"])["product"].nunique()
#         unique_products_per_shopper = unique_products_per_shopper.to_frame("unique_products_per_shopper").reset_index()
        return unique_products_per_shopper

    def unique_categories_per_shopper(self):
        unique_categories_per_shopper = self.baskets_train.groupby(["shopper"])["category"].nunique()
#         unique_categories_per_shopper = unique_categories_per_shopper.to_frame(
#             "unique_products_per_shopper").reset_index()
        return unique_categories_per_shopper

    def ratio_of_reordered_products_per_shopper(self):
        ratio_of_reordered_products_per_shopper = (
                    self.reordered_product().groupby("shopper").sum() / self.unique_products_per_shopper()).to_frame(
            "ratio_of_reordered_products").reset_index()
        return ratio_of_reordered_products_per_shopper

    def ratio_of_reordered_categories_per_shopper(self):
        ratio_of_reordered_categories_per_shopper = (
                    self.reordered_category().groupby('shopper').sum() / self.unique_categories_per_shopper()).to_frame(
            'ratio_of_reordered_categories').reset_index()
        return ratio_of_reordered_categories_per_shopper
