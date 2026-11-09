from pathlib import Path
import re
import shutil
import pandas as pd


# Ścieżki cache oraz identyfikator zbioru źródłowego
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "phishing_urls.csv"
PREPARED_DATA_PATH = DATA_DIR / "prepared_urls.csv"
KAGGLE_DATASET = "ndarvind/phiusiil-phishing-url-dataset"

URL_COLUMN_NAMES = {"url", "urls", "website"}
LABEL_COLUMN_NAMES = {"label", "type", "status", "result", "class", "phishing"}
PHISHING_LABELS = {"bad", "malicious", "phishing", "phish", "suspicious"}
LEGITIMATE_LABELS = {"good", "benign", "legitimate", "safe", "legitimate url"}
PREPARED_COLUMNS = [
    "URL", "is_phishing", "url_length", "num_dots", "num_hyphens", "num_at", "num_digits", "has_https", "has_ip",
]
IP_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")


def _find_source_columns(columns):
    # Rozpoznaje nazwy kolumn URL i etykiet niezależnie od wielkości liter
    normalized = {str(column).strip().lower(): column for column in columns}
    url_column = next(
        (normalized[name] for name in URL_COLUMN_NAMES if name in normalized), None
    )
    label_column = next(
        (normalized[name] for name in LABEL_COLUMN_NAMES if name in normalized), None
    )
    return url_column, label_column


def _download_raw_dataset():
    import kagglehub

    # Pobiera zbiór i zachowuje lokalną kopię CSV z wymaganymi kolumnami
    download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    for csv_path in sorted(download_path.rglob("*.csv")):
        columns = pd.read_csv(csv_path, nrows=0).columns
        url_column, label_column = _find_source_columns(columns)
        if url_column is not None and label_column is not None:
            RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, RAW_DATA_PATH)
            return

    raise FileNotFoundError(
        f"Nie znaleziono odpowiedniego pliku csv"
    )


def _normalize_label(value):
    # Normalizuje źródłowe etykiety do 1 = phishing oraz 0 = prawdziwy URL
    label = str(value).strip().lower()
    if label in PHISHING_LABELS:
        return 1
    if label in LEGITIMATE_LABELS:
        return 0

    numeric_label = pd.to_numeric(label, errors="coerce")
    if numeric_label == 0:
        return 1
    if numeric_label == 1:
        return 0
    return pd.NA


def _prepare_dataset(raw_df):
    # Czyści dane, mapuje etykiety i wylicza cechy strukturalne URL-i
    url_column, label_column = _find_source_columns(raw_df.columns)
    if url_column is None or label_column is None:
        raise ValueError(
            "Zbiór musi zawierać kolumnę URL oraz kolumnę etykiety "
        )

    prepared = raw_df[[url_column, label_column]].copy()
    prepared.columns = ["URL", "source_label"]
    prepared = prepared.dropna(subset=["URL", "source_label"])
    prepared["URL"] = prepared["URL"].astype(str).str.strip()
    prepared = prepared[prepared["URL"] != ""]
    prepared["is_phishing"] = prepared["source_label"].map(_normalize_label)
    prepared = prepared.dropna(subset=["is_phishing"])
    prepared["is_phishing"] = prepared["is_phishing"].astype(int)
    prepared = prepared.drop(columns="source_label")
    prepared = prepared.drop_duplicates(subset=["URL"]).reset_index(drop=True)

    urls = prepared["URL"].str.lower()
    prepared["url_length"] = urls.str.len()
    prepared["num_dots"] = urls.str.count(r"\.")
    prepared["num_hyphens"] = urls.str.count("-")
    prepared["num_at"] = urls.str.count("@")
    prepared["num_digits"] = urls.map(lambda url: sum(char.isdigit() for char in url))
    prepared["has_https"] = urls.str.startswith("https").astype(int)
    prepared["has_ip"] = urls.map(lambda url: int(bool(IP_RE.search(url))))
    prepared = prepared[PREPARED_COLUMNS]

    if prepared.empty or prepared["is_phishing"].nunique() != 2:
        raise ValueError(
            "Etykieta 0 oznacza phishing, a 1 adres legalny."
        )

    return prepared


def load_and_preprocess_dataset():
    # W pierwszej kolejności używa przygotowanego cache; pobiera dane tylko w razie potrzeby
    if PREPARED_DATA_PATH.exists():
        prepared = pd.read_csv(PREPARED_DATA_PATH)
        missing_columns = set(PREPARED_COLUMNS) - set(prepared.columns)
        if missing_columns:
            raise ValueError(
                f"Brak niektórych kolumn, usuń ten plik, aby wygenerować go ponownie."
            )
        return prepared[PREPARED_COLUMNS]

    if not RAW_DATA_PATH.exists():
        print(f"Pobieranie zbioru Kaggle: {KAGGLE_DATASET}")
        _download_raw_dataset()

    prepared = _prepare_dataset(pd.read_csv(RAW_DATA_PATH))
    PREPARED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(PREPARED_DATA_PATH, index=False)
    print(f"Przygotowane dane zapisano w: {PREPARED_DATA_PATH}")
    print(f"Liczba próbek: {len(prepared)}")
    print(f"Rozkład klas:\n{prepared['is_phishing'].value_counts().sort_index()}")
    return prepared


if __name__ == "__main__":
    load_and_preprocess_dataset()
