"""Testy samego generatora: czy dane są poprawne i zgodne z threat modelem."""

from urllib.parse import urlsplit

import pytest

from synthetic_urls import (
    BRANDS,
    LEGIT_TECHNIQUES,
    PHISHING_TECHNIQUES,
    generate_dataset,
    to_phishing_site_urls_format,
    to_phiusiil_format,
)


def test_same_seed_gives_same_data():
    a = generate_dataset(n=500, seed=7)
    b = generate_dataset(n=500, seed=7)
    assert a.equals(b)


def test_different_seed_gives_different_data():
    a = generate_dataset(n=500, seed=1)
    b = generate_dataset(n=500, seed=2)
    assert set(a["url"]) != set(b["url"])


def test_size_and_class_balance(synthetic_df):
    assert len(synthetic_df) == 4000
    assert synthetic_df["is_phishing"].sum() == 2000


def test_phishing_ratio_is_respected():
    df = generate_dataset(n=1000, seed=3, phishing_ratio=0.2)
    assert df["is_phishing"].sum() == 200


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.5])
def test_invalid_phishing_ratio_rejected(ratio):
    with pytest.raises(ValueError):
        generate_dataset(n=100, phishing_ratio=ratio)


def test_urls_are_unique(synthetic_df):
    assert synthetic_df["url"].is_unique


def test_urls_are_parseable_ascii(synthetic_df):
    for url in synthetic_df["url"]:
        parts = urlsplit(url)
        assert parts.scheme in ("http", "https"), url
        assert parts.hostname, url
        assert url.isascii(), url


def test_every_technique_present_with_correct_label(synthetic_df):
    by_label = synthetic_df.groupby("technique")["is_phishing"].unique()
    for t in PHISHING_TECHNIQUES:
        assert list(by_label[t]) == [1], t
    for t in LEGIT_TECHNIQUES:
        assert list(by_label[t]) == [0], t


def test_phishing_targets_banks(synthetic_df):
    """Threat model: przeciwnik podszywa się pod banki.

    Techniki, które zachowują nazwę marki w URL-u, muszą ją faktycznie zawierać.
    Typosquat/homoglyph celowo ją zniekształcają, a open_redirect ukrywa cel w parametrze.
    """
    tokens = [t for t, _ in BRANDS]
    brand_keeping = {"subdomain_abuse", "combosquat", "ip_host", "at_sign", "free_hosting"}
    rows = synthetic_df[synthetic_df["technique"].isin(brand_keeping)]
    for url in rows["url"]:
        assert any(t in url for t in tokens), url


def test_typosquat_and_homoglyph_never_hit_real_bank_domain(synthetic_df):
    real_hosts = {d for _, d in BRANDS} | {f"www.{d}" for _, d in BRANDS}
    rows = synthetic_df[synthetic_df["technique"].isin({"typosquat", "homoglyph"})]
    for url in rows["url"]:
        assert urlsplit(url).hostname not in real_hosts, url


def test_legit_bank_urls_use_real_domains(synthetic_df):
    real_hosts = {d for _, d in BRANDS} | {f"www.{d}" for _, d in BRANDS}
    rows = synthetic_df[synthetic_df["technique"] == "real_bank"]
    for url in rows["url"]:
        assert urlsplit(url).hostname in real_hosts, url


def test_phiusiil_format_matches_main_py_input(synthetic_df):
    out = to_phiusiil_format(synthetic_df)
    assert list(out.columns) == ["URL", "label"]
    # PhiUSIIL: 1 = legalny, 0 = phishing
    assert (out["label"] == 1 - synthetic_df["is_phishing"]).all()


def test_phishing_site_urls_format_matches_kaggle(synthetic_df):
    out = to_phishing_site_urls_format(synthetic_df)
    assert list(out.columns) == ["URL", "Label"]
    assert set(out["Label"]) == {"good", "bad"}
    assert (out.loc[synthetic_df["is_phishing"] == 1, "Label"] == "bad").all()
