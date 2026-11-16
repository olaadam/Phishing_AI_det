"""Testy end-to-end istniejących skryptów na danych z generatora.

Nie wymagają pobierania datasetu z Kaggle ani dostępu do sieci.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from synthetic_urls import to_phishing_site_urls_format, to_phiusiil_format

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_main_py_runs_end_to_end(synthetic_df, tmp_path):
    """Uruchamia main.py bez zmian, podkładając syntetyczny plik PhiUSIIL w katalogu roboczym."""
    shutil.copy(REPO_ROOT / "main.py", tmp_path / "main.py")
    to_phiusiil_format(synthetic_df).to_csv(tmp_path / "PhiUSIIL_Phishing_URL_Dataset.csv", index=False)

    result = subprocess.run(
        [sys.executable, "main.py"], cwd=tmp_path, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr

    f1_scores = [float(x) for x in re.findall(r"F1 Score: ([0-9.]+)", result.stdout)]
    fprs = [float(x) for x in re.findall(r"False Positive Rate \(FPR\): ([0-9.]+)", result.stdout)]
    # 2 modele x 2 zbiory (testowy, walidacyjny)
    assert len(f1_scores) == 4, result.stdout
    assert len(fprs) == 4, result.stdout
    assert all(0.0 <= v <= 1.0 for v in f1_scores + fprs)

    # Test poprawności pipeline'u, a nie jakości modelu: przy pomieszanych etykietach
    # lub zepsutych cechach F1 spadłoby w okolice 0.5. Dane syntetyczne są łatwiejsze
    # od prawdziwych, więc wynik nie dowodzi spełnienia kryterium F1 > 0.92 / FPR < 2%.
    assert min(f1_scores) > 0.9, result.stdout


def test_load_prepare_data_with_synthetic_kaggle_file(synthetic_df, tmp_path, monkeypatch):
    """load_prepare_data.py z podmienionym pobieraniem z Kaggle (kagglehub.dataset_download)."""
    import load_prepare_data

    to_phishing_site_urls_format(synthetic_df).to_csv(tmp_path / "phishing_site_urls.csv", index=False)
    monkeypatch.setattr(load_prepare_data.kagglehub, "dataset_download", lambda _name: str(tmp_path))

    X_train, X_test, y_train, y_test = load_prepare_data.load_and_preprocess_dataset()

    n = len(synthetic_df)
    assert X_train.shape[0] + X_test.shape[0] == n
    assert X_test.shape[0] == pytest.approx(0.2 * n, abs=1)
    assert X_train.shape[1] == X_test.shape[1]
    # stratify=y: odsetek "bad" taki sam w obu częściach
    assert (y_train == "bad").mean() == pytest.approx((y_test == "bad").mean(), abs=0.01)
