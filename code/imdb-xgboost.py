#!/usr/bin/env python

import read_imdb
import argparse, csv, os, sys, random, yaml
from datetime import datetime
from nltk.tokenize import word_tokenize
import xgboost
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

script_version = "1.0.0"


def train_models(num_runs, run_name, use_gpu):

    imdb_data = read_imdb.LoadIMDBData(seed=42)

    train_df = imdb_data.train_df
    validation_df = imdb_data.validation_df
    test_df = imdb_data.test_df

    # initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)

    # fit the count on trainig data review
    vectorizer_train = vectorizer.fit_transform(train_df.review_text)

    # transform the validation and test data
    vectorizer_val = vectorizer.transform(validation_df.review_text)
    vectorizer_test = vectorizer.transform(test_df.review_text)

    for x in range(num_runs):

        # Select a random seed every run because the default seed is 0
        seed = random.randint(0, 999999999)

        if use_gpu:
            tree_method = "gpu_hist"
        else:
            tree_method = "auto"

        model = xgboost.XGBClassifier(
            tree_method=tree_method,
            max_depth=7,
            eta=0.7,
            objective="binary:logistic",
            n_estimators=300,
            eval_metric="auc",
            seed=seed,
            # The following parameters introduce randomness
            subsample=0.8,  # Randomize
            colsample_bytree=0.8,  # Randomize
            colsample_bylevel=0.8,  # Randomize
            colsample_bynode=0.8,  # Randomize
            # OR
            # booster="gblinear",
            # feature_selector="shuffle", # Randomize
        )

        model.fit(
            vectorizer_train,
            train_df.sentiment,
            verbose=False,
            eval_set=[(vectorizer_val, validation_df.sentiment)],
        )

        predictions = model.predict(vectorizer_test)

        # calculate accuracy
        test_accuracy = metrics.accuracy_score(test_df.sentiment, predictions)

        print(f"Accuracy: {test_accuracy}")

        #
        # Write results to CSV
        #
        csv_file = os.path.basename(sys.argv[0]).split(".")[0] + ".csv"
        write_header = False

        if not os.path.isfile(csv_file):
            write_header = True

        with open(csv_file, "a") as csvfile:
            fieldnames = [
                "run_name",
                "script_version",
                "date_time",
                "python_version",
                "xgboost_version",
                "seed",
                "test_accuracy",
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            writer.writerow(
                {
                    "run_name": run_name,
                    "script_version": script_version,
                    "date_time": datetime.now(),
                    "python_version": sys.version.replace("\n", ""),
                    "xgboost_version": xgboost.__version__,
                    "seed": seed,
                    "test_accuracy": test_accuracy,
                }
            )


def parse_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-runs",
        dest="num_runs",
        help="Number of training runs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--run-name",
        dest="run_name",
        help="Name of training run",
        default="",
    )

    parser.add_argument(
        "--use-gpu", dest="use_gpu", help="Use GPU for training", action="store_true"
    )

    return parser.parse_args(args)


def get_system_info():
    if os.path.exists("system_info.py"):
        import system_info

        sysinfo = system_info.get_system_info()
        base_name = os.path.basename(sys.argv[0]).split(".")[0]

        with open("%s_system_info.yaml" % base_name, "w") as system_info_file:
            yaml.dump(sysinfo, system_info_file, default_flow_style=False)

        return sysinfo
    else:
        return None


if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])

    get_system_info()

    train_models(args.num_runs, args.run_name, args.use_gpu)
