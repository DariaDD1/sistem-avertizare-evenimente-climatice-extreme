import argparse
from pathlib import Path
import pandas as pd

FOLDER_REZULTATE = Path("outputs")
FOLDER_REZULTATE.mkdir(exist_ok=True)

COLOANE_TRASATURI = [
    "temp_aer_min",
    "temp_aer_medie",
    "temp_aer_max",
    "viteza_vant_max",
    "an",
    "luna",
    "zi_din_an",
    "saptamana",
    "anotimp",
    "variatie_temp",
    "precip_lag1",
    "precip_lag2",
    "precip_lag3",
    "precip_lag7",
    "precip_sum_3zile",
    "precip_sum_7zile",
    "precip_sum_14zile",
    "precip_medie_3zile",
    "precip_medie_7zile",
    "precip_medie_14zile",
    "temp_medie_3zile",
    "temp_medie_7zile",
    "temp_medie_14zile",
    "temp_std_3zile",
    "temp_std_7zile",
    "temp_std_14zile",
    "vant_medie_3zile",
    "vant_medie_7zile",
    "vant_medie_14zile",
    "vant_lag1",
    "vant_lag2",
    "vant_lag3",
    "ploaie_ieri",
    "zile_ploioase_consecutive",
    "zi_calduroasa",
    "zi_foarte_calduroasa",
    "zi_rece",
    "zi_vant_puternic",
    "zi_ploaie_abundenta",
    "ploaie_persistenta",
]

# Incărcare și preprocesare date meteorologice din fișiere Excel/CSV


def incarca_un_fisier(cale_fisier):
    cale_fisier = Path(cale_fisier)
    if not cale_fisier.exists():
        raise FileNotFoundError(f"Fisierul nu exista: {cale_fisier}")
    if cale_fisier.suffix.lower() in [".xlsx", ".xls"]:
        date = pd.read_excel(cale_fisier)
    elif cale_fisier.suffix.lower() == ".csv":
        date = pd.read_csv(cale_fisier)
    else:
        raise ValueError(f"Format neacceptat: {cale_fisier.suffix}")
    date["sursa_fisier"] = cale_fisier.name
    return date


def incarca_fisierele(cai_fisiere):
    return pd.concat(
        [incarca_un_fisier(cale) for cale in cai_fisiere], ignore_index=True
    )


# Funcție principală de încărcare și preprocesare a datelor, cu inginerie de trăsături și creare de indicatori pentru evenimente climatice extreme


def incarca_si_pregateste_datele(cai_fisiere):
    if isinstance(cai_fisiere, (str, Path)):
        cai_fisiere = [cai_fisiere]

    date = incarca_fisierele(cai_fisiere)
    date.columns = date.columns.str.strip()

    coloane_obligatorii = [
        "data",
        "temp_aer_min",
        "temp_aer_medie",
        "temp_aer_max",
        "viteza_vant_max",
        "cant_precipitatii",
    ]
    coloane_lipsa = [
        coloana for coloana in coloane_obligatorii if coloana not in date.columns
    ]
    if coloane_lipsa:
        raise ValueError(f"Lipsesc urmatoarele coloane obligatorii: {coloane_lipsa}")

    date["data"] = pd.to_datetime(date["data"], errors="coerce")
    date = date.dropna(subset=["data"]).sort_values("data").reset_index(drop=True)
    date["cant_precipitatii"] = date["cant_precipitatii"].clip(lower=0)

    for coloana in ["temp_aer_min", "temp_aer_medie", "temp_aer_max"]:
        date.loc[(date[coloana] <= -50) | (date[coloana] >= 60), coloana] = pd.NA

    date = date.dropna(subset=coloane_obligatorii).reset_index(drop=True)

    date["an"] = date["data"].dt.year
    date["luna"] = date["data"].dt.month
    date["zi_din_an"] = date["data"].dt.dayofyear
    date["saptamana"] = date["data"].dt.isocalendar().week.astype(int)
    date["anotimp"] = date["luna"] % 12 // 3 + 1
    date["variatie_temp"] = date["temp_aer_max"] - date["temp_aer_min"]

    for lag in [1, 2, 3, 7]:
        date[f"precip_lag{lag}"] = date["cant_precipitatii"].shift(lag)

    for fereastra in [3, 7, 14]:
        date[f"precip_sum_{fereastra}zile"] = (
            date["cant_precipitatii"].shift(1).rolling(fereastra).sum()
        )
        date[f"precip_medie_{fereastra}zile"] = (
            date["cant_precipitatii"].shift(1).rolling(fereastra).mean()
        )
        date[f"temp_medie_{fereastra}zile"] = (
            date["temp_aer_medie"].shift(1).rolling(fereastra).mean()
        )
        date[f"temp_std_{fereastra}zile"] = (
            date["temp_aer_medie"].shift(1).rolling(fereastra).std()
        )
        date[f"vant_medie_{fereastra}zile"] = (
            date["viteza_vant_max"].shift(1).rolling(fereastra).mean()
        )

    for lag in [1, 2, 3]:
        date[f"vant_lag{lag}"] = date["viteza_vant_max"].shift(lag)

    date["ploaie_ieri"] = (date["cant_precipitatii"].shift(1) > 0).astype(int)
    ploaie_anterioara = (date["cant_precipitatii"].shift(1) > 0).astype(int)
    grupuri_uscate = (date["cant_precipitatii"].shift(1).fillna(0) == 0).cumsum()
    date["zile_ploioase_consecutive"] = ploaie_anterioara.groupby(
        grupuri_uscate
    ).cumsum()

    prag_precip_90 = date["cant_precipitatii"].quantile(0.90)
    prag_temp_max_90 = date["temp_aer_max"].quantile(0.90)
    prag_temp_max_95 = date["temp_aer_max"].quantile(0.95)
    prag_temp_min_10 = date["temp_aer_min"].quantile(0.10)
    prag_vant_90 = date["viteza_vant_max"].quantile(0.90)

    date["zi_calduroasa"] = (date["temp_aer_max"] >= prag_temp_max_90).astype(int)
    date["zi_foarte_calduroasa"] = (date["temp_aer_max"] >= prag_temp_max_95).astype(
        int
    )
    date["zi_rece"] = (date["temp_aer_min"] <= prag_temp_min_10).astype(int)
    date["zi_vant_puternic"] = (date["viteza_vant_max"] >= prag_vant_90).astype(int)
    date["zi_ploaie_abundenta"] = (date["cant_precipitatii"] >= prag_precip_90).astype(
        int
    )
    date["ploaie_persistenta"] = (
        date["precip_sum_7zile"] >= date["precip_sum_7zile"].quantile(0.90)
    ).astype(int)

    date["scor_risc"] = (
        2 * date["zi_ploaie_abundenta"]
        + 2 * date["ploaie_persistenta"]
        + date["zi_vant_puternic"]
        + date["zi_foarte_calduroasa"]
        + date["zi_rece"]
        + (date["zile_ploioase_consecutive"] >= 3).astype(int)
    )

    date["eveniment_extrem"] = (date["scor_risc"] >= 2).astype(int)
    date = date.dropna().reset_index(drop=True)
    date.head(5).to_csv(
        FOLDER_REZULTATE / "primele_5_inregistrari_cu_features.csv", index=False
    )
    return date


# Rulare directa a preprocesarii datelor din linia de comanda

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", nargs="+", required=True)
    argumente = parser.parse_args()

    date_pregatite = incarca_si_pregateste_datele(argumente.data_path)
    date_pregatite.to_csv(
        FOLDER_REZULTATE / "date_meteorologice_procesate.csv", index=False
    )
    print("Preprocesarea datelor a fost finalizata.")
