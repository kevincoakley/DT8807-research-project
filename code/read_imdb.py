#!/usr/bin/env python

#
# This script is used to load the IMDB data into a Pandas dataframe
# and split it into train, validation, and test sets.
#
# last updated: kevincoakley (03/08/2023)
#

import os, nltk, re, string, urllib.request, tarfile, shutil
from collections import defaultdict
import pandas as pd

nltk.download("punkt")


class LoadIMDBData(object):
    def __init__(self, path="./", seed=None):
        self.path = path
        self.imdb_path = os.path.join(self.path, "aclImdb/")
        self.train_path = os.path.join(self.imdb_path, "train/")
        self.test_path = os.path.join(self.imdb_path, "test/")
        self.seed = seed
        self.train_df, self.validation_df = self.train_data()
        self.test_df = self.test_data()

    def download_reviews(self):
        """
        Download the IMDB reviews from the internet.
        """
        # If the path doesn't exist, download the data
        if not os.path.isdir(self.imdb_path):
            urllib.request.urlretrieve(
                "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                "aclImdb_v1.tar.gz",
            )
            # Untar aclImdb_v1.tar.gz
            imdb_tar = tarfile.open("aclImdb_v1.tar.gz")
            imdb_tar.extractall(self.path)
            imdb_tar.close()
            # Remove the unspervised data
            shutil.rmtree(os.path.join(self.imdb_path, "train/unsup"))

    def clean_text(self, text):
        # make all text lowercase
        text = text.lower()
        # replace <br /> with space
        text = text.replace("<br />", " ")
        # remove all punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

        return text

    def read_reviews(self, path):
        """
        Read the data from the path and return a pandas DataFrame with
        columns for the file name, review text, and sentiment.
        """

        reviews = defaultdict(list)

        # Read the data by walking the path
        for root, dirs, files in os.walk(path):
            for file in files:
                # Only read the text files
                if file.endswith(".txt"):
                    # Only read the files numbers and _ are in the file name
                    if file.split(".")[0].replace("_", "").isdigit():
                        file_path = os.path.join(root, file)

                        # Add the file name, review text, and sentiment to the reviews
                        with open(file_path, "r") as file_open:
                            # Get the file name from file_path
                            reviews["file_name"].append(file_path)
                            # Get the review text from reading the open file
                            reviews["review_text"].append(file_open.read())
                            # Get the sentiment from the file_path
                            if "pos" in file_path:
                                reviews["sentiment"].append("pos")
                            elif "neg" in file_path:
                                reviews["sentiment"].append("neg")
                            else:
                                reviews["sentiment"].append("ERROR")

        reviews_pd = pd.DataFrame(reviews)

        # Convert the sentiment column to 0 and 1
        reviews_pd.sentiment = reviews_pd.sentiment.apply(
            lambda x: 1 if x == "pos" else 0
        )

        # Clean the review text
        reviews_pd.loc[:, "review_text"] = reviews_pd.review_text.apply(self.clean_text)

        return reviews_pd

    def randomize_data(self, df, seed=None):
        """
        Randomize the rows of data.
        """
        # Randomize the rows of data
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    def split_data(self, df, split_percent=0.8):
        """
        Split the data into a training and validation set.
        """

        validation_df = (
            df[int(df.shape[0] * split_percent) :].copy().reset_index(drop=True)
        )
        train_df = df[: int(df.shape[0] * split_percent)].copy().reset_index(drop=True)

        # Check that the split_train_df and validation_df have the same number
        # of rows as the original train_df
        assert len(train_df) + len(validation_df) == len(df)

        # Check that the split_train_df and validation_df have no overlapping rows
        assert len(set(train_df.file_name).intersection(validation_df.file_name)) == 0

        return train_df, validation_df

    def train_data(self):
        """
        Returns the training data.
        """
        self.download_reviews()
        x = self.read_reviews(self.train_path)
        x = self.randomize_data(x, self.seed)
        x, y = self.split_data(x)
        return x, y

    def test_data(self):
        """
        Returns the test data.
        """
        self.download_reviews()
        x = self.read_reviews(self.test_path)
        return x
