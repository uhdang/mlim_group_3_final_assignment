from collections import Counter

from lightgbm import LGBMClassifier


class Trainer:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def define_lgb_clf(self):
        counter = Counter(self.y_train)
        class_imbalance = counter[0] / counter[1]
        lgb_clf = LGBMClassifier(scale_pos_weight=class_imbalance)
        return lgb_clf

    def fit_model(self):
        categorical = self.X_train.select_dtypes(exclude=np.number).columns.tolist()
        lgb_clf = self.define_lgb_clf()
        lgb_clf.fit(self.X_train, self.y_train, categorical_feature=categorical)
        return lgb_clf

    # def predict(self):


