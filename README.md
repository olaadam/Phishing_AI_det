# ZTKiK_phishing
Wykrywanie Ataków Phishingowych i Złośliwych URL-i za Pomocą TF-IDF i Naive Bayes

## Uruchomienie

Zainstaluj zależności poleceniem `pip install -r requirements.txt`, a następnie
uruchom `python main.py`.

`load_prepare_data.py` pobiera zbiór `ndarvind/phiusiil-phishing-url-dataset`
z Kaggle tylko wtedy, gdy nie ma lokalnej kopii. Oczyszczone dane wraz z etykietą
i cechami strukturalnymi zapisuje do `data/prepared_urls.csv`. `main.py` korzysta
z tego przygotowanego pliku. Etykieta `is_phishing` ma wartość `1` dla phishingu
i `0` dla prawdziwych adresów. Przy pierwszym pobraniu KaggleHub może wymagać
konfiguracji dostępu do Kaggle.
