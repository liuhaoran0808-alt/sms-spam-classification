import io
import logging
import os
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


@dataclass
class ProjectConfig:
    data_url: str = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00228/smsspamcollection.zip"
    )
    data_file: str = "SMSSpamCollection"
    max_workers_thread: int = 4
    max_workers_process: int = os.cpu_count() or 2
    random_seed: int = 42
    sample_multiplier: int = 5


config = ProjectConfig()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Download and load the SMS Spam Collection dataset."""
    logger.info("Downloading dataset from UCI Repository...")

    response = requests.get(config.data_url, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall()

    df = pd.read_csv(config.data_file, sep="\t", names=["label", "message"])

    # Repeat the dataset to make processing-time comparison easier to observe.
    df = pd.concat([df] * config.sample_multiplier, ignore_index=True)
    logger.info("Dataset loaded successfully. Total records: %s", len(df))
    return df


def clean_text(text: Optional[str]) -> str:
    """Clean one SMS message for vectorization."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def measure_processing(raw_texts: list[str]) -> tuple[list[str], dict[str, float]]:
    """Compare sequential, multithreading, and multiprocessing text cleaning."""
    times: dict[str, float] = {}

    logger.info("--- Starting Sequential Processing ---")
    start_time = time.perf_counter()
    seq_cleaned = [clean_text(text) for text in raw_texts]
    times["Sequential"] = time.perf_counter() - start_time

    logger.info("--- Starting Multithreading Processing ---")
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.max_workers_thread) as executor:
        thread_cleaned = list(executor.map(clean_text, raw_texts))
    times["Multithreading"] = time.perf_counter() - start_time

    logger.info("--- Starting Multiprocessing Processing ---")
    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=config.max_workers_process) as executor:
        process_cleaned = list(executor.map(clean_text, raw_texts))
    times["Multiprocessing"] = time.perf_counter() - start_time

    # Use multiprocessing result for the ML pipeline, matching the original notebook.
    if process_cleaned != thread_cleaned or process_cleaned != seq_cleaned:
        logger.warning("Processing methods produced different cleaned outputs.")

    for method, duration in times.items():
        logger.info("%s Execution Time: %.4f seconds", method, duration)

    return process_cleaned, times


def train_classifier(cleaned_texts: list[str], labels: pd.Series) -> float:
    """Train and evaluate a TF-IDF + Naive Bayes spam classifier."""
    logger.info("Starting Machine Learning Pipeline...")

    vectorizer = TfidfVectorizer(max_features=5000)
    features = vectorizer.fit_transform(cleaned_texts)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=config.random_seed,
    )

    model = MultinomialNB()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info("Model Training Complete. Accuracy: %.2f%%", accuracy * 100)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy


def plot_results(times: dict[str, float], accuracy: float) -> None:
    """Show execution time and model accuracy charts."""
    methods = list(times.keys())
    durations = list(times.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    bars = ax1.bar(
        methods,
        durations,
        color=["#ff9999", "#66b3ff", "#99ff99"],
        edgecolor="black",
        alpha=0.8,
    )
    ax1.set_ylabel("Time Taken (Seconds)")
    ax1.set_title("Execution Time Comparison")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax2.bar(["Model Accuracy"], [accuracy], color="gold", edgecolor="black", width=0.4)
    ax2.set_ylabel("Accuracy (0-1)")
    ax2.set_title("Final Model Accuracy")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.text(
        0,
        accuracy + 0.02,
        f"{accuracy * 100:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        color="darkorange",
    )

    plt.tight_layout()
    plt.show()


def main() -> None:
    logger.info("System initialized. Configuration loaded.")
    df = load_data()
    print(df.head())

    raw_texts = df["message"].tolist()
    cleaned_texts, times = measure_processing(raw_texts)
    accuracy = train_classifier(cleaned_texts, df["label"])
    plot_results(times, accuracy)


if __name__ == "__main__":
    main()
