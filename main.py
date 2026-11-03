import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from scipy.sparse import csr_matrix
import kagglehub
import glob

# zbiór danych z Kaggle
path = kagglehub.dataset_download("ndarvind/phiusiil-phishing-url-dataset")
print("Path to dataset files:", path)
csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
print("Znalezione pliki CSV:", csv_files)

# Załaduj dane
df = pd.read_csv(csv_files[0])
df = df.dropna(subset=['URL', 'label']).reset_index(drop=True)
df["URL"] = df["URL"].astype(str)

#podział 1 = legalny, 0 = phishing
df['is_phishing'] = (df['label'].astype(int) == 0).astype(int)

df_trainval, df_test = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df['is_phishing'])
df_train, df_val = train_test_split(
    df_trainval, test_size=1/7, random_state=42, stratify=df_trainval['is_phishing'])

print(f"Train: {len(df_train)/len(df):.1%} | Val: {len(df_val)/len(df):.1%} | Test: {len(df_test)/len(df):.1%}")

y_train = df_train['is_phishing'].values
y_val = df_val['is_phishing'].values
y_test = df_test['is_phishing'].values

X_text_train = df_train['URL'].values
X_text_val = df_val['URL'].values
X_text_test = df_test['URL'].values

# slowa kluczowe
DEFAULT_KEYWORDS = [
    "login", "signin", "secure", "verify", "account", "update", "bank", "confirm",
    "password", "webscr", "billing", "wallet", "support", "security", "paypal",
    "ebay", "apple", "microsoft", "amazon", "alert", "suspend", "unlock",
]

EXTERNAL_KEYWORDS = [] #jakby był plik to sie doda

# słowa wyuczone z danych 
token_vec = CountVectorizer(token_pattern=r"[a-zA-Z]{3,}", binary=True, min_df=20)
X_tok_train = token_vec.fit_transform(X_text_train)
chi_scores, _ = chi2(X_tok_train, y_train)
chi_scores = np.nan_to_num(chi_scores)
tokens = np.array(token_vec.get_feature_names_out())
pos_rate = np.asarray(X_tok_train[y_train == 1].mean(axis=0)).ravel()
neg_rate = np.asarray(X_tok_train[y_train == 0].mean(axis=0)).ravel()
top_idx = np.argsort(-(chi_scores * (pos_rate > neg_rate)))[:50]
LEARNED_KEYWORDS = tokens[top_idx].tolist()
print("Wyuczone słowa:", LEARNED_KEYWORDS[:15])

ALL_KEYWORDS = sorted(set(EXTERNAL_KEYWORDS) | set(LEARNED_KEYWORDS))

# cechy strukturalne (długość, liczba kropek, myślników, cyfr, @, https, IP, słowa kluczowe)
IP_RE = re.compile(r"\d{1,3}(\.\d{1,3}){3}")

def extract_features(url):
    u = url.lower()
    return [
        len(u),                                        # url_length
        u.count('.'),                                  # num_dots
        u.count('-'),                                  # num_hyphens
        u.count('@'),
        sum(c.isdigit() for c in u),
        int(u.startswith('https')),
        int(bool(IP_RE.search(u))),
        sum(u.count(k) for k in EXTERNAL_KEYWORDS),    # słowa kluczowe (zewnętrzne)
        sum(u.count(k) for k in LEARNED_KEYWORDS),     # słowa kluczowe (wyuczone)
        sum(k in u for k in ALL_KEYWORDS),             # liczba różnych słów kluczowych
    ]

X_num_train = np.array([extract_features(u) for u in X_text_train])
X_num_val = np.array([extract_features(u) for u in X_text_val])
X_num_test = np.array([extract_features(u) for u in X_text_test])

# fit tylko na train
scaler = MinMaxScaler()
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)

X_train = hstack([vectorizer.fit_transform(X_text_train),
                  csr_matrix(scaler.fit_transform(X_num_train))]).tocsr()
X_val = hstack([vectorizer.transform(X_text_val),
                csr_matrix(np.clip(scaler.transform(X_num_val), 0, 1))]).tocsr()
X_test = hstack([vectorizer.transform(X_text_test),
                 csr_matrix(np.clip(scaler.transform(X_num_test), 0, 1))]).tocsr()

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb_test = nb_model.predict(X_test)
y_pred_nb_val = nb_model.predict(X_val)

svm_model = LinearSVC(random_state=42, max_iter=10000, dual=False)
svm_model.fit(X_train, y_train)
y_pred_svm_test = svm_model.predict(X_test)
y_pred_svm_val = svm_model.predict(X_val)


def print_metrics(y_true, y_pred, model_name, dataset_name):
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"Model: {model_name} | Zbiór: {dataset_name}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print("Raport klasyfikacji:")
    print(classification_report(y_true, y_pred))
    print("\n")


print_metrics(y_test, y_pred_nb_test, "Naive Bayes", "Testowy (30%)")
print_metrics(y_val, y_pred_nb_val, "Naive Bayes", "Walidacyjny (10%)")
print_metrics(y_test, y_pred_svm_test, "SVM", "Testowy (30%)")
print_metrics(y_val, y_pred_svm_val, "SVM", "Walidacyjny (10%)")