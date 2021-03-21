def prepare_data():
    from sklearn.metrics import roc_auc_score
    import pickle
    import os

    has_cv_dict_available = input("Do you have generated data available in data folder as pickle? (Yes / No) ")
    if has_cv_dict_available == "Yes":
        print("Loading...")
        # cv_dict_name = input("Please provide name of the datafile with extension i.e. cv_dict.pickle ")
        cv_dict_name = "cv_dict.pickle"
        # Load Data From Picklek
        with open(os.getcwd() + "/data/" + cv_dict_name, 'rb') as f:
            cv_dict = pickle.load(f)
    else:
        cv_dict = generate_data()

    prediction_best_params_available = input("Do you have prediction and best_params available? (Yes / No) ")
    if prediction_best_params_available == "Yes":
        print("Loading...")
        # Load Data From Picklek
        with open(os.getcwd() + "/data/best_param.pickle", 'rb') as f:
            best_params = pickle.load(f)
        with open(os.getcwd() + "/data/pred.pickle", 'rb') as f:
            prediction = pickle.load(f)
    else:
        lgb_clf = hyper_tuning(cv_dict)
        prediction = lgb_clf.predict(cv_dict['X_test'][3], num_iteration=lgb_clf.best_iteration)
        best_params = lgb_clf.params

    AUC = roc_auc_score(
        y_true = cv_dict['y_test'][3],
        y_score = prediction
    )

    print("Best params:", best_params)
    print("  Accuracy = {}".format(AUC))
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))

def hyper_tuning(cv_dict):
    import optuna.integration.lightgbm as lightgb

    dtrain = lightgb.Dataset(cv_dict['X_train'][0], label=cv_dict['y_train'][0])
    X_test0, y_test0 = downsample(cv_dict['X_test'][0], cv_dict['y_test'][0])
    dval0 = lightgb.Dataset(X_test0, label=y_test0)
    X_test1, y_test1 = downsample(cv_dict['X_test'][1], cv_dict['y_test'][1])
    dval1 = lightgb.Dataset(X_test1, label=y_test1)
    X_test2, y_test2 = downsample(cv_dict['X_test'][2], cv_dict['y_test'][2])
    dval2 = lightgb.Dataset(X_test2, label=y_test2)
    X_test3, y_test3 = downsample(cv_dict['X_test'][3], cv_dict['y_test'][3])
    dval3 = lightgb.Dataset(X_test3, label=y_test3)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    lgb_clf = lightgb.train(
        params,
        dtrain,
        categorical_feature=['shopper', 'product', 'category', 'coupon', 'coupon_in_same_category'],
        valid_sets=[dval0, dval1, dval2],
        verbose_eval=100,
        early_stopping_rounds=100
    )
    return lgb_clf



def downsample(df, y):
    import pandas as pd
    pd.options.mode.chained_assignment = None  # default='warn'
    from sklearn.utils import resample

    df_target_coupon = df.loc[(y==1) | (df['coupon']=='Yes')]
    y_target_coupon = y.loc[(y==1) | (df['coupon']=='Yes')]
    df_down = df.loc[(y==0) & (df['coupon']=='No')]
    y_down = y.loc[(y==0) & (df["coupon"]=="No")]
    df_down, y_down = resample(
        df_down,
        y_down,
        replace=False,
        n_samples=df_target_coupon.shape[0],
        stratify=df_down['shopper']
    )
    df_all = pd.concat([df_target_coupon, df_down], ignore_index=True)
    y_all = pd.concat([y_target_coupon, y_down], ignore_index=True)

    return df_all, y_all

def generate_data():
    from tqdm import tqdm
    from dataloader import Dataloader
    from dataloader import create_combined_dict
    import os

    n_shoppers = 10_000
    weeks = [86, 87, 88, 89]
    shopper_list = list(range(2000))
    shopper_chunks = [shopper_list[i:i + 100] for i in range(0, len(shopper_list), 100)]

    cv_dict = {
        'X_train': list(),
        'y_train': list(),
        'X_test': list(),
        'y_test': list()
    }

    path = os.getcwd() + "data/"
    data = Dataloader(path=path)

    data.create_category_table(n_shoppers=n_shoppers)

    for i, week in enumerate(weeks):
        print(f"week: {week}")

        X_train_list = list()
        y_train_list = list()
        X_test_list = list()
        y_test_list = list()

        for idx, shopper in enumerate(tqdm(shopper_chunks)):
            # print(f"shopper_chunk index: {idx}")

            # train-test-split
            data.train_test_split(week, shopper)

            # data add categories
            data.add_categories()

            # create features
            data.create_feature_dict()

            # combine everything
            if i == 0:
                X_train, y_train, X_test, y_test = data.make_featured_data()
                X_train_list.append(X_train)
                y_train_list.append(y_train)
                del X_train, y_train
            else:
                _, _, X_test, y_test = data.make_featured_data()

            X_test_list.append(X_test)
            y_test_list.append(y_test)
            del X_test, y_test

        cv_dict = create_combined_dict(X_train_list, y_train_list, X_test_list, y_test_list, cv_dict)
    return cv_dict

if __name__ == "__main__":
    print("Running generate_data")
    prepare_data()

