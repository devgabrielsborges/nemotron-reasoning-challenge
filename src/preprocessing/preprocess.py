import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def preprocess_data(data: pd.DataFrame, target_column: str | None = None):
    target_column = target_column or os.getenv("TARGET_COLUMN")
    prompt_column = os.getenv("PROMPT_COLUMN", "prompt")
    n_components = int(os.getenv("SVD_COMPONENTS", "256"))

    required_columns = {prompt_column, target_column}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in train data: {sorted(missing_columns)}"
        )

    X_text = data[prompt_column].fillna("").astype(str)
    y = data[target_column].astype(str)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
    )
    X_train_sparse = vectorizer.fit_transform(X_train_text)
    X_test_sparse = vectorizer.transform(X_test_text)

    if X_train_sparse.shape[1] >= 3:
        effective_components = max(
            2,
            min(n_components, X_train_sparse.shape[1] - 1, X_train_sparse.shape[0] - 1),
        )
        reducer = TruncatedSVD(n_components=effective_components, random_state=42)
        X_train_preprocessed = reducer.fit_transform(X_train_sparse)
        X_test_preprocessed = reducer.transform(X_test_sparse)
    else:
        X_train_preprocessed = X_train_sparse.toarray()
        X_test_preprocessed = X_test_sparse.toarray()

    output_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train_preprocessed.npy", X_train_preprocessed)
    np.save(output_dir / "X_test_preprocessed.npy", X_test_preprocessed)
    np.save(output_dir / "y_train.npy", y_train.values)
    np.save(output_dir / "y_test.npy", y_test.values)

    raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
    submission_test_path = raw_dir / "test.csv"
    if submission_test_path.exists():
        submission_data = pd.read_csv(submission_test_path)
        if prompt_column not in submission_data.columns:
            raise ValueError(f"Missing required column in test data: {prompt_column}")

        X_submission_text = submission_data[prompt_column].fillna("").astype(str)
        X_submission_sparse = vectorizer.transform(X_submission_text)
        if X_train_sparse.shape[1] >= 3:
            X_submission_preprocessed = reducer.transform(X_submission_sparse)
        else:
            X_submission_preprocessed = X_submission_sparse.toarray()
        np.save(output_dir / "X_submission_preprocessed.npy", X_submission_preprocessed)


if __name__ == "__main__":
    load_dotenv(override=True)
    data = pd.read_csv(Path(os.getenv("DATA_RAW_DIR", "data/raw")) / "train.csv")
    preprocess_data(data)
    print("Data preprocessed successfully")
