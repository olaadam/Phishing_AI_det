# ZTKiK_phishing

Wykrywanie ataków phishingowych i złośliwych URL-i za pomocą TF-IDF i Naive Bayes
(dla porównania także liniowy SVM). Model patrzy wyłącznie na tekst adresu URL,
nie pobiera stron.

## Zawartość repo

| Ścieżka | Co to jest |
|---|---|
| `load_prepare_data.py` | Pobiera dataset PhiUSIIL z Kaggle (tylko gdy nie ma lokalnej kopii), czyści go, ujednolica etykiety i liczy cechy strukturalne URL-i. Wynik zapisuje do `data/prepared_urls.csv`. |
| `main.py` | Eksperyment: wczytuje przygotowane dane przez `load_prepare_data.py`, dzieli je, buduje cechy, trenuje Naive Bayes i LinearSVC, wypisuje metryki i rysuje macierze pomyłek. |
| `data/raw/phishing_urls.csv` | Surowa kopia datasetu PhiUSIIL (235 795 wierszy, 55 kolumn). Jest w repo. |
| `data/prepared_urls.csv` | Przygotowane dane (235 370 wierszy, 9 kolumn). Jest w repo. |
| `tests/synthetic_urls.py` | Generator syntetycznych URL-i (legalnych i phishingowych) do testów i demo. |
| `tests/test_synthetic_urls.py` | Testy generatora. |
| `tests/test_pipeline_on_synthetic.py` | Testy `load_prepare_data.py` i `main.py` na danych syntetycznych. |
| `tests/conftest.py` | Konfiguracja pytest (ścieżki importu, wspólny zbiór 4000 URL-i). |
| `requirements.txt` | Zależności do uruchamiania (pandas, scikit-learn, kagglehub, matplotlib). |
| `requirements-dev.txt` | To samo + scipy, Faker, pytest (do testów). |

## 1. Instalacja

Wymagany Python 3.10+ (sprawdzone na 3.12).

Standardowy `venv` + `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt     # albo requirements.txt, jeśli nie potrzebujesz testów
```

Albo `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

Dalsze komendy zakładają aktywne środowisko i uruchamianie z katalogu głównego repo.

## 2. Uruchomienie

```bash
python main.py
```

To wystarczy. `main.py` sam woła `load_prepare_data.py`, więc nie trzeba niczego
uruchamiać wcześniej. Pełny przebieg na 235 tys. URL-i trwa około 20 s. Na końcu
otwiera się okno z macierzami pomyłek. Program kończy się po jego zamknięciu.

Bez okna (np. przez SSH albo w CI):

```bash
MPLBACKEND=Agg python main.py
```

Samo przygotowanie danych, bez trenowania:

```bash
python load_prepare_data.py
```

## 3. Co się dzieje krok po kroku

### 3.1 `load_prepare_data.py`: skąd biorą się dane

Funkcja `load_and_preprocess_dataset()` sprawdza po kolei:

1. **Jest `data/prepared_urls.csv`?** Wczytuje go, sprawdza, czy ma wszystkie
   wymagane kolumny, i zwraca. Nic więcej się nie dzieje.
2. **Jest `data/raw/phishing_urls.csv`?** Przygotowuje z niego dane (krok 4).
3. **Nie ma żadnego pliku?** Pobiera dataset `ndarvind/phiusiil-phishing-url-dataset`
   przez `kagglehub` (do `~/.cache/kagglehub/`). Spośród pobranych plików CSV wybiera
   pierwszy (alfabetycznie), który ma kolumnę URL i kolumnę etykiety, i kopiuje go do
   `data/raw/phishing_urls.csv`. Jeśli Kaggle odmówi pobrania, skonfiguruj token API
   (`~/.kaggle/kaggle.json` albo zmienne `KAGGLE_USERNAME` i `KAGGLE_KEY`).
4. **Przygotowanie** (`_prepare_dataset`):
   - znajduje kolumny po nazwie, bez względu na wielkość liter: URL to `url`/`urls`/`website`,
     etykieta to `label`/`type`/`status`/`result`/`class`/`phishing`. Pozostałe ~50 kolumn
     PhiUSIIL jest pomijanych;
   - usuwa wiersze bez URL-a albo etykiety, obcina spacje, usuwa puste URL-e;
   - ujednolica etykietę do kolumny **`is_phishing`: 1 = phishing, 0 = legalny**.
     Tekstowe `bad/malicious/phishing/phish/suspicious` → 1, `good/benign/legitimate/safe` → 0,
     liczbowe w konwencji PhiUSIIL `0` → 1 i `1` → 0. Inne wartości są odrzucane;
   - usuwa duplikaty URL-i (w PhiUSIIL to 425 wierszy, wszystkie phishingowe);
   - liczy cechy strukturalne na URL-u zamienionym na małe litery:

     | Kolumna | Znaczenie |
     |---|---|
     | `url_length` | długość URL-a |
     | `num_dots` | liczba `.` |
     | `num_hyphens` | liczba `-` |
     | `num_at` | liczba `@` |
     | `num_digits` | liczba cyfr |
     | `has_https` | 1, jeśli URL zaczyna się od `https` |
     | `has_ip` | 1, jeśli gdziekolwiek w URL-u jest coś w formacie IPv4 |

   - przerywa z błędem, jeśli po czyszczeniu nie zostały obie klasy;
   - zapisuje wynik do `data/prepared_urls.csv` i wypisuje liczbę próbek i rozkład klas.

Po przygotowaniu: 134 850 URL-i legalnych i 100 520 phishingowych.

**Uwaga o cache:** jeśli zmienisz coś w `_prepare_dataset` (np. dodasz cechę) albo podmienisz
`data/raw/phishing_urls.csv`, usuń `data/prepared_urls.csv`. Bez tego zmiany nie zostaną użyte,
bo pierwszy krok zwraca stary plik.

### 3.2 `main.py`: trening i ocena

1. **Podział danych** ze stratyfikacją po `is_phishing` (`random_state=42`):
   najpierw 30% na test, potem z reszty 1/7 na walidację, co daje
   **60% trening / 10% walidacja / 30% test**. Proporcje są wypisywane na początku.
2. **Wyuczone słowa kluczowe** (tylko na zbiorze treningowym):
   - `CountVectorizer` wycina z URL-i słowa z co najmniej 3 liter, występujące w ≥ 20 URL-ach;
   - dla każdego słowa liczony jest test chi² względem etykiety;
   - zostają słowa częstsze w phishingu niż w legalnych URL-ach, z nich 50 o najwyższym chi².
     Pierwsze 15 jest wypisywane, np. `http, web, app, firebaseapp, html, php, ipfs, ...`.
3. **Cechy słów kluczowych** dla każdego URL-a:
   - suma wystąpień słów z listy ręcznej `DEFAULT_KEYWORDS` (`login`, `verify`, `paypal`, ...);
   - suma wystąpień słów wyuczonych;
   - liczba różnych słów z obu list, które pojawiają się w URL-u.
4. **Macierz cech**:
   - TF-IDF na n-gramach znakowych 2–4 (max 5000 cech);
   - 7 cech strukturalnych z `load_prepare_data.py` i 3 cechy słów kluczowych,
     przeskalowane `MinMaxScaler` do [0, 1].

   TF-IDF i skaler są dopasowywane tylko na treningu. Walidacja i test używają `transform`,
   a wartości cech liczbowych są przycinane do [0, 1], bo `MultinomialNB` nie przyjmuje
   wartości ujemnych.
5. **Modele**: `MultinomialNB` oraz `LinearSVC(dual=False, max_iter=10000)`.
6. **Metryki** dla każdego modelu na walidacji i teście. Klasa pozytywna to **phishing (1)**:
   - F1 dla klasy phishing;
   - FPR = FP / (FP + TN), czyli odsetek legalnych URL-i oznaczonych jako phishing;
   - pełny `classification_report`.
7. **Wykres**: macierze pomyłek obu modeli na zbiorze testowym (TN/FP/FN/TP z liczbami i %).

Wyniki na prawdziwych danych (zbiór testowy):

| Model | F1 | FPR |
|---|---|---|
| Naive Bayes | 0.9816 | 0.0068 |
| LinearSVC | 0.9974 | 0.0001 |

## 4. Testy

Testy nie potrzebują internetu ani konta Kaggle i nie zmieniają plików w `data/`.
Wszystkie dane generuje `tests/synthetic_urls.py`.

```bash
pytest                                       # wszystkie testy (24, kilka sekund)
pytest -v                                    # z nazwami testów
pytest tests/test_synthetic_urls.py          # tylko generator
pytest tests/test_pipeline_on_synthetic.py   # tylko load_prepare_data.py i main.py
pytest -k cache                              # testy, których nazwa zawiera "cache"
```

### Testy generatora (`test_synthetic_urls.py`)

Sprawdzają:
- powtarzalność dla tego samego seeda i różne dane dla różnych seedów;
- proporcje klas i odrzucanie złych wartości `phishing_ratio`;
- unikalność URL-i, schemat http/https, niepusty host, tylko ASCII;
- obecność każdej techniki ataku z właściwą etykietą;
- to, że techniki zachowujące markę faktycznie zawierają nazwę banku;
- to, że typosquat/homoglif nigdy nie trafia w prawdziwą domenę banku;
- zgodność formatów eksportu.

### Testy pipeline'u (`test_pipeline_on_synthetic.py`)

W testach `load_prepare_data.py` ścieżki `RAW_DATA_PATH` i `PREPARED_DATA_PATH` są
przekierowane do katalogu tymczasowego, a `kagglehub.dataset_download` jest podmienione
na katalog z syntetycznym CSV:

| Test | Co sprawdza |
|---|---|
| `test_download_prepare_and_cache` | Bez plików lokalnych następuje pobranie (dokładnie raz, z właściwą nazwą datasetu). Plik CSV bez kolumny etykiety jest pomijany. Oba pliki cache są zapisane, kolumny się zgadzają, a etykiety PhiUSIIL są poprawnie odwrócone (`label 0` → `is_phishing 1`). |
| `test_structural_features_match_techniques` | Cechy pasują do technik z generatora: `ip_host` → `has_ip=1`, `at_sign` → `num_at≥1`, `real_bank` → `has_https=1`, a długość i liczba kropek są policzone poprawnie. |
| `test_good_bad_labels_are_supported` | Etykiety tekstowe `good`/`bad` też są mapowane na 0/1. |
| `test_cleaning_drops_duplicates_empty_and_unknown_labels` | Duplikaty, puste URL-e, brak URL-a i nieznana etykieta są usuwane. |
| `test_existing_raw_file_skips_download` | Gdy jest `data/raw/phishing_urls.csv`, nic nie jest pobierane. |
| `test_existing_prepared_file_is_used_as_is` | Gdy jest `data/prepared_urls.csv`, jest zwracany bez pobierania i bez surowego pliku. |
| `test_prepared_file_with_missing_columns_is_rejected` | Niekompletny `prepared_urls.csv` daje `ValueError`. |
| `test_main_py_runs_end_to_end` | Kopiuje `main.py` i `load_prepare_data.py` do katalogu tymczasowego, wkłada tam syntetyczny `data/raw/phishing_urls.csv` i uruchamia `main.py` jako osobny proces z `MPLBACKEND=Agg`. Sprawdza kod wyjścia 0, zapis `prepared_urls.csv`, podział 60/10/30, wypisanie wyuczonych słów, 4 wartości F1 i 4 FPR w [0, 1] oraz F1 > 0.9. |

Próg F1 > 0.9 ma wyłapać zepsuty pipeline (np. pomieszane etykiety dałyby F1 około 0.5).
Wynik na danych syntetycznych (F1 ~0.99) nie dowodzi spełnienia kryterium
F1 > 0.92 / FPR < 2%, bo dane syntetyczne są łatwiejsze od prawdziwych.

## 5. Generator syntetycznych URL-i

```bash
python tests/synthetic_urls.py -n 2000 --format raw -o synthetic.csv                 # url,is_phishing,technique
python tests/synthetic_urls.py -n 2000 --format phiusiil -o synthetic.csv            # URL,label (1 = legalny), jak surowy PhiUSIIL
python tests/synthetic_urls.py -n 2000 --format phishing-site-urls -o synthetic.csv  # URL,Label (good/bad)
```

Opcje: `-n` liczba URL-i (domyślnie 2000), `--seed` (domyślnie 42),
`--phishing-ratio` udział phishingu (domyślnie 0.5, musi być w (0, 1)).

Dokładnie te dane, na których lecą testy:

```bash
python tests/synthetic_urls.py -n 4000 --seed 42 --format raw -o urls_z_testow.csv
```

Losowe fragmenty (słowa, domeny, IP, hashe) dostarcza biblioteka [Faker](https://faker.readthedocs.io).
Logika ataków i lista banków są zdefiniowane w `tests/synthetic_urls.py`.

Techniki phishingu (kolumna `technique` w formacie `raw`, `is_phishing = 1`):

| Technika | Jak powstaje | Przykład |
|---|---|---|
| `typosquat` | jedna litera w nazwie banku usunięta, podwojona, zamieniona z sąsiednią albo podmieniona | `http://pkopb.pl/update/index.php?session=47bcd733b8b1a` |
| `homoglyph` | litera zamieniona na podobny znak (`o→0, l→1, i→1, e→3, a→4, s→5, m→rn`) | `https://1ng.pl/potwierdz` |
| `subdomain_abuse` | prawdziwa domena jako subdomena obcej | `http://www.paypal.com.rasa-hundred.com/account/secure` |
| `combosquat` | marka sklejona ze słowem kluczowym, podejrzana końcówka domeny | `http://aktualizacja.santander.tk/update/verify?session=553898901` |
| `ip_host` | publiczny IP (czasem z portem), marka w ścieżce | `http://145.22.42.33:8080/ipko/account/token/doplata.php?session=...` |
| `at_sign` | prawdziwa domena przed `@`, faktyczny host po `@` | `http://www.aliorbank.pl@niebieskiuklad.com/weryfikacja/doplata/aktualizacja` |
| `free_hosting` | marka w subdomenie darmowego hostingu | `https://pekao-result301.netlify.app/potwierdz.aspx` |
| `open_redirect` | zwykła domena z parametrem przekierowania na stronę przeciwnika | `https://jones.com/out?next=http%3A%2F%2Fcommon-maintain.shop/weryfikacja` |

Legalne URL-e (`is_phishing = 0`): `faker_generic` (losowe strony z Fakera, np.
`http://johnson.com/mouth-play`) i `real_bank` (prawdziwe domeny banków, zawsze `https`,
także ze ścieżkami `login`/`logowanie`, np. `https://ipko.pl/firmy/kredyty/ago-site-face`).
