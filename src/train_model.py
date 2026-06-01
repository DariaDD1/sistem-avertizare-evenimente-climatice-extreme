import argparse
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from preprocessing import COLOANE_TRASATURI, incarca_si_pregateste_datele

warnings.filterwarnings("ignore")
FOLDER_REZULTATE = Path("outputs")
FOLDER_REZULTATE.mkdir(exist_ok=True)

# ==============================
# MODELE SI EVALUARE
# ==============================

def obtine_modele():
    return {
        "Regresie Logistica": Pipeline([
            ("standardizare", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42, n_jobs=-1),
        "SVM": Pipeline([
            ("standardizare", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
        ]),
        "Retea Neuronala": Pipeline([
            ("standardizare", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001, learning_rate_init=0.001, max_iter=500, early_stopping=True, random_state=42)),
        ]),
    }


def evalueaza_modelul(model, date_test, tinta_test, prag_predictie=0.30):
    probabilitati = model.predict_proba(date_test)[:, 1]
    predictii = (probabilitati >= prag_predictie).astype(int)
    matrice_confuzie = confusion_matrix(tinta_test, predictii, labels=[0, 1])
    adevarat_negativ, fals_pozitiv, fals_negativ, adevarat_pozitiv = matrice_confuzie.ravel()

    indicatori = {
        "acuratete": accuracy_score(tinta_test, predictii),
        "acuratete_echilibrata": balanced_accuracy_score(tinta_test, predictii),
        "precizie": precision_score(tinta_test, predictii, zero_division=0),
        "recall": recall_score(tinta_test, predictii, zero_division=0),
        "scor_f1": f1_score(tinta_test, predictii, zero_division=0),
        "roc_auc": roc_auc_score(tinta_test, probabilitati),
        "mcc": matthews_corrcoef(tinta_test, predictii),
        "prag_predictie": prag_predictie,
        "adevarat_negativ": int(adevarat_negativ),
        "fals_pozitiv": int(fals_pozitiv),
        "fals_negativ": int(fals_negativ),
        "adevarat_pozitiv": int(adevarat_pozitiv),
    }

    raport_clasificare = classification_report(tinta_test, predictii, zero_division=0)
    return indicatori, matrice_confuzie, raport_clasificare

# ==============================
# ANTRENARE SI SELECTIE MODEL OPTIM
# ==============================

def compara_modelele(date):
    rezultate = []
    modele_antrenate = {}
    impartiri_temporale = {"80/20": 0.80, "70/30": 0.70}

    if "eveniment_extrem" not in date.columns:
        raise ValueError("Coloana 'eveniment_extrem' lipseste.")

    for nume_impartire, proportie_antrenare in impartiri_temporale.items():
        index_impartire = int(len(date) * proportie_antrenare)
        date_antrenare = date.iloc[:index_impartire].copy()
        date_testare = date.iloc[index_impartire:].copy()

        trasaturi_antrenare = date_antrenare[COLOANE_TRASATURI]
        tinta_antrenare = date_antrenare["eveniment_extrem"]
        trasaturi_testare = date_testare[COLOANE_TRASATURI]
        tinta_testare = date_testare["eveniment_extrem"]

        print("\n" + "=" * 70)
        print(f"IMPARTIRE TEMPORALA: {nume_impartire}")
        print(f"Randuri antrenare: {len(trasaturi_antrenare)}")
        print(f"Randuri testare: {len(trasaturi_testare)}")
        print("=" * 70)

        for nume_model, model in obtine_modele().items():
            print(f"\nAntrenez modelul: {nume_model}")
            model.fit(trasaturi_antrenare, tinta_antrenare)
            indicatori, matrice_confuzie, raport_clasificare = evalueaza_modelul(model, trasaturi_testare, tinta_testare, prag_predictie=0.30)

            print(f"Recall: {indicatori['recall']:.4f}")
            print(f"Scor F1: {indicatori['scor_f1']:.4f}")
            print(f"ROC-AUC: {indicatori['roc_auc']:.4f}")
            print("Matricea de confuzie:")
            print(matrice_confuzie)
            print("Raport de clasificare:")
            print(raport_clasificare)

            rezultate.append({"impartire": nume_impartire, "model": nume_model, **indicatori})
            modele_antrenate[(nume_impartire, nume_model)] = {"model": model, "indicatori": indicatori}

    tabel_rezultate = pd.DataFrame(rezultate).sort_values(by=["recall", "scor_f1", "roc_auc"], ascending=False).reset_index(drop=True)
    tabel_rezultate.to_csv(FOLDER_REZULTATE / "rezultate_comparare_modele.csv", index=False)

    cel_mai_bun_rand = tabel_rezultate.iloc[0]
    informatii_model_optim = modele_antrenate[(cel_mai_bun_rand["impartire"], cel_mai_bun_rand["model"])]

    print("\nMODEL RECOMANDAT PENTRU DASHBOARD")
    print(f"Impartire: {cel_mai_bun_rand['impartire']}")
    print(f"Model: {cel_mai_bun_rand['model']}")
    print(f"Recall: {cel_mai_bun_rand['recall']:.4f}")

    return tabel_rezultate, informatii_model_optim, cel_mai_bun_rand

# ==============================
# SALVARE REZULTATE
# ==============================

def salveaza_rezultatele(date, informatii_model_optim, cel_mai_bun_rand):
    joblib.dump(informatii_model_optim["model"], FOLDER_REZULTATE / "model_optim.pkl")

    with open(FOLDER_REZULTATE / "coloane_trasaturi.json", "w", encoding="utf-8") as fisier:
        json.dump(COLOANE_TRASATURI, fisier, indent=4)

    metadate = {
        "cea_mai_buna_impartire": cel_mai_bun_rand["impartire"],
        "cel_mai_bun_model": cel_mai_bun_rand["model"],
        "regula_selectie": "Cel mai mare Recall, apoi Scor F1, apoi ROC-AUC",
        "prag_predictie": float(cel_mai_bun_rand["prag_predictie"]),
    }

    with open(FOLDER_REZULTATE / "metadate_model_optim.json", "w", encoding="utf-8") as fisier:
        json.dump(metadate, fisier, indent=4)

    date.to_csv(FOLDER_REZULTATE / "date_meteorologice_procesate.csv", index=False)
    print("Fisierele necesare dashboardului au fost salvate in folderul outputs.")


def antreneaza_si_salveaza(cai_fisiere):
    date = incarca_si_pregateste_datele(cai_fisiere)
    comparatie, informatii_model_optim, cel_mai_bun_rand = compara_modelele(date)
    salveaza_rezultatele(date, informatii_model_optim, cel_mai_bun_rand)
    return comparatie

# ==============================
# RULARE DIRECTA
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", nargs="+", required=True)
    argumente = parser.parse_args()
    antreneaza_si_salveaza(argumente.data_path)
