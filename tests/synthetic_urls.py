"""Generator syntetycznych adresów URL (legalnych i phishingowych) do testów.

Dane są generowane zgodnie z threat modelem projektu ("Phishing Email Evasion"):
przeciwnik tworzy domeny podszywające się pod banki. Każdy URL phishingowy
dostaje kolumnę ``technique`` z nazwą użytej techniki, żeby można było
sprawdzić, które ataki model wykrywa, a które przepuszcza.

Losowe fragmenty (słowa, ścieżki, adresy IP, domeny) pochodzą z biblioteki
Faker. Generator jest deterministyczny dla danego ``seed`` i wersji Fakera.

Użycie jako skrypt (eksport CSV, np. do ręcznego sprawdzenia lub demo):

    python tests/synthetic_urls.py -n 2000 --format phiusiil -o synthetic.csv
"""

from __future__ import annotations

import argparse
import random
import unicodedata

import pandas as pd
from faker import Faker

# Banki/serwisy płatnicze, pod które podszywa się przeciwnik.
# (token marki, prawdziwa domena)
BRANDS = [
    ("pkobp", "pkobp.pl"),
    ("ipko", "ipko.pl"),
    ("mbank", "mbank.pl"),
    ("santander", "santander.pl"),
    ("ing", "ing.pl"),
    ("pekao", "pekao.com.pl"),
    ("millennium", "bankmillennium.pl"),
    ("alior", "aliorbank.pl"),
    ("paypal", "paypal.com"),
    ("revolut", "revolut.com"),
]

# Prawdziwe ścieżki na stronach banków. Zawierają też "login"/"logowanie",
# żeby słowo kluczowe samo w sobie nie zdradzało etykiety.
LEGIT_BANK_PATHS = [
    "",
    "kontakt",
    "klienci-indywidualni/konta",
    "klienci-indywidualni/karty",
    "firmy/kredyty",
    "o-banku/aktualnosci",
    "logowanie",
    "login",
    "pomoc/bezpieczenstwo",
    "oferta/lokaty",
]

PHISHING_KEYWORDS = [
    "login", "logowanie", "verify", "weryfikacja", "secure", "account",
    "konto", "update", "aktualizacja", "blokada", "doplata", "sesja",
    "token", "auth", "potwierdz", "odblokuj",
]

ATTACKER_TLDS = ["xyz", "top", "online", "info", "ru", "cn", "tk", "click", "shop", "site", "com", "net"]

FREE_HOSTING = ["000webhostapp.com", "weebly.com", "firebaseapp.com", "netlify.app", "glitch.me", "duckdns.org"]

HOMOGLYPHS = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5", "m": "rn"}

PHISHING_TECHNIQUES = [
    "typosquat",
    "homoglyph",
    "subdomain_abuse",
    "combosquat",
    "ip_host",
    "at_sign",
    "free_hosting",
    "open_redirect",
]

LEGIT_TECHNIQUES = ["faker_generic", "real_bank"]


class SyntheticUrlGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.fake = Faker(["pl_PL", "en_US"])
        self.fake.seed_instance(seed)

    # ---------- pomocnicze ----------

    def _scheme(self, https_prob: float) -> str:
        return "https" if self.rng.random() < https_prob else "http"

    def _brand(self) -> tuple[str, str]:
        return self.rng.choice(BRANDS)

    def _word(self) -> str:
        """Losowe słowo z Fakera bez polskich znaków (nazwy hostów muszą być ASCII)."""
        word = self.fake.word().lower().replace("ł", "l")
        return unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode()

    def _attacker_domain(self) -> str:
        return f"{self._word()}{self.rng.choice(['', '-', '24', 'pl'])}{self._word()}.{self.rng.choice(ATTACKER_TLDS)}"

    def _phish_path(self) -> str:
        parts = self.rng.sample(PHISHING_KEYWORDS, k=self.rng.randint(1, 3))
        path = "/".join(parts)
        if self.rng.random() < 0.5:
            path += self.rng.choice([".php", ".html", ".aspx", "/index.php"])
        if self.rng.random() < 0.5:
            path += f"?session={self.fake.sha1()[: self.rng.randint(8, 32)]}"
        return path

    @staticmethod
    def _split_domain(domain: str) -> tuple[str, str]:
        name, _, tld = domain.partition(".")
        return name, tld

    # ---------- URL-e legalne ----------

    def legit_faker_generic(self) -> str:
        url = self.fake.url().rstrip("/")
        roll = self.rng.random()
        if roll < 0.4:
            url += "/" + self.fake.uri_path()
        elif roll < 0.7:
            url += "/" + self.fake.slug()
        if self.rng.random() < 0.2:
            url += f"?{self._word()}={self.rng.randint(1, 9999)}"
        return url

    def legit_real_bank(self) -> str:
        _, domain = self._brand()
        host = domain if self.rng.random() < 0.3 else f"www.{domain}"
        url = f"https://{host}/{self.rng.choice(LEGIT_BANK_PATHS)}"
        roll = self.rng.random()
        if roll < 0.4:
            url = url.rstrip("/") + "/" + self.fake.slug()
        elif roll < 0.7:
            url += f"?utm_source={self.rng.choice(['newsletter', 'google', 'facebook'])}&utm_campaign={self.fake.slug()}"
        return url

    # ---------- URL-e phishingowe ----------

    def phish_typosquat(self) -> str:
        _, domain = self._brand()
        name, tld = self._split_domain(domain)
        typo = name
        while typo == name:  # "replace" może wylosować tę samą literę
            chars = list(name)
            i = self.rng.randrange(len(chars))
            op = self.rng.choice(["drop", "double", "swap", "replace"])
            if op == "drop" and len(chars) > 3:
                del chars[i]
            elif op == "swap" and i < len(chars) - 1:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif op == "replace":
                chars[i] = self.rng.choice("abcdefghijklmnopqrstuvwxyz")
            else:
                chars.insert(i, chars[i])
            typo = "".join(chars)
        return f"{self._scheme(0.6)}://{typo}.{tld}/{self._phish_path()}"

    def phish_homoglyph(self) -> str:
        _, domain = self._brand()
        name, tld = self._split_domain(domain)
        candidates = [i for i, c in enumerate(name) if c in HOMOGLYPHS]
        if candidates:
            i = self.rng.choice(candidates)
            name = name[:i] + HOMOGLYPHS[name[i]] + name[i + 1 :]
        else:
            name += "-pl"
        return f"{self._scheme(0.7)}://{name}.{tld}/{self._phish_path()}"

    def phish_subdomain_abuse(self) -> str:
        _, domain = self._brand()
        return f"{self._scheme(0.7)}://www.{domain}.{self._attacker_domain()}/{self._phish_path()}"

    def phish_combosquat(self) -> str:
        token, _ = self._brand()
        kw = self.rng.choice(PHISHING_KEYWORDS)
        sep = self.rng.choice(["-", "", "."])
        first, second = (token, kw) if self.rng.random() < 0.5 else (kw, token)
        return f"{self._scheme(0.5)}://{first}{sep}{second}.{self.rng.choice(ATTACKER_TLDS)}/{self._phish_path()}"

    def phish_ip_host(self) -> str:
        token, _ = self._brand()
        port = self.rng.choice(["", ":8080", ":8443", ":81"])
        return f"http://{self.fake.ipv4_public()}{port}/{token}/{self._phish_path()}"

    def phish_at_sign(self) -> str:
        _, domain = self._brand()
        return f"{self._scheme(0.5)}://www.{domain}@{self._attacker_domain()}/{self._phish_path()}"

    def phish_free_hosting(self) -> str:
        token, _ = self._brand()
        sub = f"{token}{self.rng.choice(['-', ''])}{self._word()}{self.rng.randint(1, 999)}"
        return f"https://{sub}.{self.rng.choice(FREE_HOSTING)}/{self._phish_path()}"

    def phish_open_redirect(self) -> str:
        """Legalnie wyglądający host z parametrem przekierowania na stronę przeciwnika."""
        host = self.fake.domain_name()
        target = f"{self._attacker_domain()}/{self.rng.choice(PHISHING_KEYWORDS)}"
        param = self.rng.choice(["url", "redirect", "next", "goto", "r"])
        return f"https://{host}/out?{param}=http%3A%2F%2F{target}"

    # ---------- zbiór danych ----------

    def generate(self, n: int, phishing_ratio: float = 0.5) -> pd.DataFrame:
        """Zwraca DataFrame z kolumnami: url, is_phishing (0/1), technique.

        Techniki są przydzielane cyklicznie (duplikat przechodzi do następnej
        techniki), więc każda pojawia się w zbiorze przy n >= 2 * liczba
        technik. URL-e są unikalne.
        """
        if not 0 < phishing_ratio < 1:
            raise ValueError("phishing_ratio musi być w przedziale (0, 1)")
        n_phish = round(n * phishing_ratio)
        n_legit = n - n_phish

        rows: list[tuple[str, int, str]] = []
        seen: set[str] = set()

        def fill(count: int, techniques: list[str], prefix: str, label: int) -> None:
            produced, attempts = 0, 0
            while produced < count:
                attempts += 1
                if attempts > count * 50:
                    raise RuntimeError(f"Nie udało się wygenerować {count} unikalnych URL-i ({prefix})")
                technique = techniques[attempts % len(techniques)]
                url = getattr(self, f"{prefix}_{technique}")()
                if url in seen:
                    continue
                seen.add(url)
                rows.append((url, label, technique))
                produced += 1

        fill(n_legit, LEGIT_TECHNIQUES, "legit", 0)
        fill(n_phish, PHISHING_TECHNIQUES, "phish", 1)

        df = pd.DataFrame(rows, columns=["url", "is_phishing", "technique"])
        return df.sample(frac=1, random_state=self.rng.randint(0, 2**31 - 1)).reset_index(drop=True)


def generate_dataset(n: int = 2000, seed: int = 42, phishing_ratio: float = 0.5) -> pd.DataFrame:
    return SyntheticUrlGenerator(seed).generate(n, phishing_ratio)


def to_phiusiil_format(df: pd.DataFrame) -> pd.DataFrame:
    """Format pliku PhiUSIIL_Phishing_URL_Dataset.csv czytanego przez main.py.

    W PhiUSIIL label = 1 oznacza URL legalny, label = 0 phishing.
    """
    return pd.DataFrame({"URL": df["url"], "label": 1 - df["is_phishing"]})


def to_phishing_site_urls_format(df: pd.DataFrame) -> pd.DataFrame:
    """Format Kaggle taruntiwarihp/phishing-site-urls czytany przez load_prepare_data.py (URL, Label: good/bad)."""
    return pd.DataFrame({"URL": df["url"], "Label": df["is_phishing"].map({0: "good", 1: "bad"})})


FORMATS = {
    "raw": lambda df: df,
    "phiusiil": to_phiusiil_format,
    "phishing-site-urls": to_phishing_site_urls_format,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=2000, help="liczba URL-i (domyślnie 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phishing-ratio", type=float, default=0.5)
    parser.add_argument("--format", choices=FORMATS, default="raw")
    parser.add_argument("-o", "--output", required=True, help="ścieżka pliku CSV")
    args = parser.parse_args()

    df = FORMATS[args.format](generate_dataset(args.n, args.seed, args.phishing_ratio))
    df.to_csv(args.output, index=False)
    print(f"Zapisano {len(df)} wierszy do {args.output}")


if __name__ == "__main__":
    main()
