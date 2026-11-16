"""Testy end-to-end load_prepare_data.py i main.py na danych z generatora.

Nie wymagają pobierania datasetu z Kaggle ani dostępu do sieci, nie dotykają
też prawdziwego katalogu data/ w repo.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import load_prepare_data
from synthetic_urls import to_phishing_site_urls_format, to_phiusiil_format

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Przekierowuje ścieżki cache load_prepare_data.py do katalogu tymczasowego."""
    raw = tmp_path / "data" / "raw" / "phishing_urls.csv"
    prepared = tmp_path / "data" / "prepared_urls.csv"
    monkeypatch.setattr(load_prepare_data, "RAW_DATA_PATH", raw)
    monkeypatch.setattr(load_prepare_data, "PREPARED_DATA_PATH", prepared)
    return raw, prepared


@pytest.fixture
def fake_kaggle(tmp_path, monkeypatch):
    """Podmienia kagglehub.dataset_download na katalog z plikami CSV. Zwraca (katalog, lista wywołań)."""
    download_dir = tmp_path / "kaggle_download"
    download_dir.mkdir()
    calls = []

    def fake_download(name):
        calls.append(name)
        return str(download_dir)

    monkeypatch.setattr("kagglehub.dataset_download", fake_download)
    return download_dir, calls


def _assert_labels_match(prepared, synthetic_df):
    expected = synthetic_df.set_index("url")["is_phishing"]
    assert len(prepared) == len(synthetic_df)
    assert (prepared.set_index("URL")["is_phishing"] == expected.loc[prepared["URL"]]).all()


def test_download_prepare_and_cache(synthetic_df, isolated_data_dir, fake_kaggle):
    """Brak cache: pobranie (zamockowane), wybór właściwego CSV, przygotowanie i zapis obu plików."""
    raw_path, prepared_path = isolated_data_dir
    download_dir, calls = fake_kaggle
    # Plik bez kolumny etykiety ma zostać pominięty przy wyborze CSV.
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(download_dir / "a_metadata.csv", index=False)
    to_phiusiil_format(synthetic_df).to_csv(download_dir / "b_dataset.csv", index=False)

    prepared = load_prepare_data.load_and_preprocess_dataset()

    assert calls == [load_prepare_data.KAGGLE_DATASET]
    assert raw_path.exists() and prepared_path.exists()
    assert list(prepared.columns) == load_prepare_data.PREPARED_COLUMNS
    # PhiUSIIL: label 1 = legalny -> is_phishing 0, label 0 = phishing -> is_phishing 1
    _assert_labels_match(prepared, synthetic_df)
    assert pd.read_csv(prepared_path).equals(prepared)


def test_structural_features_match_techniques(synthetic_df, isolated_data_dir, fake_kaggle):
    """Cechy strukturalne są zgodne z tym, jak generator budował URL-e."""
    download_dir, _ = fake_kaggle
    to_phiusiil_format(synthetic_df).to_csv(download_dir / "dataset.csv", index=False)

    prepared = load_prepare_data.load_and_preprocess_dataset()
    merged = prepared.merge(synthetic_df, left_on="URL", right_on="url")
    by_tech = merged.groupby("technique")

    assert (by_tech.get_group("ip_host")["has_ip"] == 1).all()
    assert (by_tech.get_group("at_sign")["num_at"] >= 1).all()
    assert (by_tech.get_group("real_bank")["has_https"] == 1).all()
    assert (merged["url_length"] == merged["URL"].str.len()).all()
    assert (merged["num_dots"] == merged["URL"].str.count(r"\.")).all()


def test_good_bad_labels_are_supported(synthetic_df, isolated_data_dir, fake_kaggle):
    """Etykiety tekstowe good/bad (inne datasety z Kaggle) też są mapowane na 0/1."""
    download_dir, _ = fake_kaggle
    to_phishing_site_urls_format(synthetic_df).to_csv(download_dir / "dataset.csv", index=False)

    prepared = load_prepare_data.load_and_preprocess_dataset()

    _assert_labels_match(prepared, synthetic_df)


def test_cleaning_drops_duplicates_empty_and_unknown_labels(synthetic_df, isolated_data_dir, fake_kaggle):
    download_dir, _ = fake_kaggle
    raw = to_phiusiil_format(synthetic_df.head(100))
    dirty = pd.concat(
        [
            raw,
            raw.head(5),  # duplikaty URL-i
            pd.DataFrame({"URL": ["   ", None], "label": [1, 0]}),  # pusty URL, brak URL-a
            pd.DataFrame({"URL": ["http://unknown-label.com"], "label": [7]}),  # nieznana etykieta
        ]
    )
    dirty.to_csv(download_dir / "dataset.csv", index=False)

    prepared = load_prepare_data.load_and_preprocess_dataset()

    assert len(prepared) == 100
    assert prepared["URL"].is_unique
    assert "http://unknown-label.com" not in set(prepared["URL"])


def test_existing_raw_file_skips_download(synthetic_df, isolated_data_dir, fake_kaggle):
    raw_path, prepared_path = isolated_data_dir
    _, calls = fake_kaggle
    raw_path.parent.mkdir(parents=True)
    to_phiusiil_format(synthetic_df).to_csv(raw_path, index=False)

    load_prepare_data.load_and_preprocess_dataset()

    assert calls == []
    assert prepared_path.exists()


def test_existing_prepared_file_is_used_as_is(synthetic_df, isolated_data_dir, fake_kaggle):
    raw_path, _ = isolated_data_dir
    download_dir, calls = fake_kaggle
    to_phiusiil_format(synthetic_df).to_csv(download_dir / "dataset.csv", index=False)
    first = load_prepare_data.load_and_preprocess_dataset()
    raw_path.unlink()
    calls.clear()

    second = load_prepare_data.load_and_preprocess_dataset()

    assert calls == []
    assert not raw_path.exists()  # surowy plik nie był potrzebny
    pd.testing.assert_frame_equal(first, second)


def test_prepared_file_with_missing_columns_is_rejected(isolated_data_dir):
    _, prepared_path = isolated_data_dir
    prepared_path.parent.mkdir(parents=True)
    pd.DataFrame({"URL": ["http://a.com"], "is_phishing": [0]}).to_csv(prepared_path, index=False)

    with pytest.raises(ValueError):
        load_prepare_data.load_and_preprocess_dataset()


def test_main_py_runs_end_to_end(synthetic_df, tmp_path):
    """Uruchamia main.py bez zmian w kopii repo z syntetycznym data/raw/phishing_urls.csv.

    Surowy plik już istnieje, więc nic nie jest pobierane z Kaggle; load_prepare_data.py
    przygotowuje dane, a main.py trenuje modele i rysuje macierze pomyłek (backend Agg,
    więc plt.show() nie otwiera okna).
    """
    for name in ("main.py", "load_prepare_data.py"):
        shutil.copy(REPO_ROOT / name, tmp_path / name)
    raw_path = tmp_path / "data" / "raw" / "phishing_urls.csv"
    raw_path.parent.mkdir(parents=True)
    to_phiusiil_format(synthetic_df).to_csv(raw_path, index=False)

    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, "main.py"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "data" / "prepared_urls.csv").exists()

    assert re.search(r"Train: 60\.0% \| Val: 10\.0% \| Test: 30\.0%", result.stdout), result.stdout
    assert "Wyuczone słowa:" in result.stdout

    f1_scores = [float(x) for x in re.findall(r"F1 Score: ([0-9.]+)", result.stdout)]
    fprs = [float(x) for x in re.findall(r"False Positive Rate \(FPR\): ([0-9.]+)", result.stdout)]
    # 2 modele x 2 zbiory (walidacyjny, testowy)
    assert len(f1_scores) == 4, result.stdout
    assert len(fprs) == 4, result.stdout
    assert all(0.0 <= v <= 1.0 for v in f1_scores + fprs)

    # Test poprawności pipeline'u, a nie jakości modelu: przy pomieszanych etykietach
    # lub zepsutych cechach F1 spadłoby w okolice 0.5. Dane syntetyczne są łatwiejsze
    # od prawdziwych, więc wynik nie dowodzi spełnienia kryterium F1 > 0.92 / FPR < 2%.
    assert min(f1_scores) > 0.9, result.stdout
