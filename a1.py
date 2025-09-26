import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def stacking_experiment(path, sample_size=5000):
    # load and shuffle
    df = pd.read_csv(path).sample(frac=1, random_state=42).reset_index(drop=True)

    # sample 
    if len(df) > sample_size:
        df = df.head(sample_size)

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # define base learners
    base_learners = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("svc", SVC(probability=True, kernel="rbf", random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("gb", GradientBoostingClassifier(random_state=42))
    ]

    # candidate meta models
    meta_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(probability=True, random_state=42)
    }

    # evaluate each meta model
    outcomes = {}
    for label, meta in meta_models.items():
        model = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta,
            cv=3,
            n_jobs=-1
        )
        # cross validation on whole dataset
        cv_scores = cross_val_score(model, X, y, cv=3, n_jobs=-1)

        # train on split
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        test_acc = accuracy_score(y_te, preds)

        outcomes[label] = {
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": test_acc
        }

    return outcomes

# run
results = stacking_experiment("DCT_mal.csv")

for meta, stats in results.items():
    print(f"\n>>> Stacking with {meta} as Meta Model")
    print("Test Accuracy:", stats["test_accuracy"])
    print("CV Accuracy:", stats["cv_mean_accuracy"], "±", stats["cv_std"])
