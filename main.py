import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from load_prepare_data import load_and_preprocess_dataset


# Wczytanie przygotowanych danych i podział na trening, walidację oraz test.
df = load_and_preprocess_dataset()
df_trainval, df_test = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["is_phishing"]
)
df_train, df_val = train_test_split(
    df_trainval, test_size=1/7, random_state=42, stratify=df_trainval['is_phishing'])

print(f"Train: {len(df_train)/len(df):.1%} | Val: {len(df_val)/len(df):.1%} | Test: {len(df_test)/len(df):.1%}")

y_train = df_train["is_phishing"].to_numpy()
y_val = df_val["is_phishing"].to_numpy()
y_test = df_test["is_phishing"].to_numpy()

X_text_train = df_train["URL"].to_numpy()
X_text_val = df_val["URL"].to_numpy()
X_text_test = df_test["URL"].to_numpy()

DEFAULT_KEYWORDS = [
    "login", "signin", "secure", "verify", "account", "update", "bank", "confirm",
    "password", "webscr", "billing", "wallet", "support", "security", "paypal",
    "ebay", "apple", "microsoft", "amazon", "alert", "suspend", "unlock",
]
EXTERNAL_KEYWORDS = DEFAULT_KEYWORDS

# Słowa charakterystyczne dla phishingu
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


def extract_keyword_features(urls):
    features = []
    for url in urls:
        normalized_url = url.lower()
        features.append(
            [
                sum(normalized_url.count(keyword) for keyword in EXTERNAL_KEYWORDS),
                sum(normalized_url.count(keyword) for keyword in LEARNED_KEYWORDS),
                sum(keyword in normalized_url for keyword in ALL_KEYWORDS),
            ]
        )
    return np.asarray(features)


# Cechy strukturalne URL-i przygotowane wcześniej przez load_prepare_data.py.
NUMERIC_COLUMNS = [
    "url_length", "num_dots", "num_hyphens", "num_at", "num_digits", "has_https", "has_ip",
]

X_num_train = np.column_stack(
    [df_train[NUMERIC_COLUMNS].to_numpy(), extract_keyword_features(X_text_train)]
)
X_num_val = np.column_stack(
    [df_val[NUMERIC_COLUMNS].to_numpy(), extract_keyword_features(X_text_val)]
)
X_num_test = np.column_stack(
    [df_test[NUMERIC_COLUMNS].to_numpy(), extract_keyword_features(X_text_test)]
)

scaler = MinMaxScaler()
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)

# TF-IDF i skalowanie są dopasowywane tylko na treningu; walidacja i test używają transform.
X_train = hstack(
    [
        vectorizer.fit_transform(X_text_train),
        csr_matrix(scaler.fit_transform(X_num_train)),
    ]
).tocsr()
X_val = hstack(
    [
        vectorizer.transform(X_text_val),
        csr_matrix(np.clip(scaler.transform(X_num_val), 0, 1)),
    ]
).tocsr()
X_test = hstack(
    [
        vectorizer.transform(X_text_test),
        csr_matrix(np.clip(scaler.transform(X_num_test), 0, 1)),
    ]
).tocsr()

# Trenowanie modeli i predykcje na zbiorach walidacyjnym oraz testowym.
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb_test = nb_model.predict(X_test)
y_pred_nb_val = nb_model.predict(X_val)

svm_model = LinearSVC(random_state=42, max_iter=10000, dual=False)
svm_model.fit(X_train, y_train)
y_pred_svm_test = svm_model.predict(X_test)
y_pred_svm_val = svm_model.predict(X_val)


# Raport metryk
def print_metrics(y_true, y_pred, model_name, dataset_name):
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"Model: {model_name} | Zbiór: {dataset_name}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print("Raport klasyfikacji:")
    print(classification_report(y_true, y_pred, zero_division=0))


print_metrics(y_val, y_pred_nb_val, "Naive Bayes", "Walidacyjny (10%)")
print_metrics(y_val, y_pred_svm_val, "SVC", "Walidacyjny (10%)")
print_metrics(y_test, y_pred_nb_test, "Naive Bayes", "Testowy (30%)")
print_metrics(y_test, y_pred_svm_test, "SVC", "Testowy (30%)")

test_confusion_matrices = [
    ("Naive Bayes", confusion_matrix(y_test, y_pred_nb_test, labels=[0, 1])),
    ("SVC", confusion_matrix(y_test, y_pred_svm_test, labels=[0, 1])),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
fig.suptitle("Macierze pomyłek — zbiór testowy", fontsize=16, fontweight="bold")
for ax, (title, cm) in zip(axes, test_confusion_matrices):
    image = ax.imshow(cm, cmap="Blues")
    ax.set(
        title=title,
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Prawdziwy", "Phishing"],
        yticklabels=["Prawdziwy", "Phishing"],
        xlabel="Przewidziana klasa",
        ylabel="Rzeczywista klasa",
    )
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    max_value = cm.max()
    for row in range(2):
        for column in range(2):
            outcome = [["TN", "FP"], ["FN", "TP"]][row][column]
            value = cm[row, column]
            percentage = value / cm.sum() * 100 if cm.sum() else 0
            text_color = "white" if value > max_value / 2 else "#172554"
            ax.text(
                column,
                row,
                f"{outcome}\n{value:,}\n({percentage:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold",
            )

    ax.set_aspect("equal")

plt.show()
