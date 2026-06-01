import json
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from alerts import trimite_alerta_email, trebuie_trimisa_alerta
from preprocessing import COLOANE_TRASATURI

FOLDER_REZULTATE = Path("outputs")
CALE_MODEL = FOLDER_REZULTATE / "model_optim.pkl"
CALE_METADATE = FOLDER_REZULTATE / "metadate_model_optim.json"
CALE_DATE = FOLDER_REZULTATE / "date_meteorologice_procesate.csv"

# ==============================
# CONFIGURARE SI INCARCARE
# ==============================

st.set_page_config(page_title="Dashboard avertizare climatica", layout="wide")
st.markdown("""
<style>
section[data-testid="stSidebar"] div[data-testid="stNumberInputContainer"] { background-color: transparent !important; }
section[data-testid="stSidebar"] div[data-baseweb="input"] { background-color: #0e1117 !important; }
section[data-testid="stSidebar"] div[data-baseweb="base-input"] { background-color: #0e1117 !important; }
section[data-testid="stSidebar"] input[data-testid="stNumberInputField"] { background-color: #0e1117 !important; }
.block-container { padding-top: 1rem; padding-bottom: 0rem; }
div[data-testid="stAlert"] { padding: 6px 10px; font-size: 13px; }
div[data-testid="stMetricLabel"] { font-size: 10px !important; }
div[data-testid="stMetricValue"] { font-size: 18px !important; }
div[data-testid="stMetric"] { padding: 2px 4px !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def incarca_modelul():
    return joblib.load(CALE_MODEL)

@st.cache_data
def incarca_metadatele():
    with open(CALE_METADATE, "r", encoding="utf-8") as fisier:
        return json.load(fisier)

@st.cache_data
def incarca_datele():
    date = pd.read_csv(CALE_DATE)
    date["data"] = pd.to_datetime(date["data"], errors="coerce")
    return date

if not CALE_MODEL.exists() or not CALE_METADATE.exists() or not CALE_DATE.exists():
    st.error("Lipsesc fisierele modelului antrenat. Ruleaza mai intai: python src/main.py --data-path data/fisier.xlsx")
    st.stop()

model = incarca_modelul()
metadate = incarca_metadatele()
date = incarca_datele()
st.title("Sistem hibrid de avertizare timpurie")

# ==============================
# SETARI SI FILTRARE
# ==============================

st.sidebar.header("Setari alerte")
prag_implicit_precipitatii = float(date["cant_precipitatii"].quantile(0.90))
prag_precipitatii = st.sidebar.number_input("Prag precipitatii (mm)", min_value=0.0, max_value=200.0, value=prag_implicit_precipitatii, step=1.0)
prag_probabilitate = st.sidebar.slider("Prag probabilitate ML", min_value=0.0, max_value=1.0, value=0.80, step=0.05)
prag_scor_risc = st.sidebar.number_input("Prag scor hibrid", min_value=0, max_value=10, value=2, step=1)

st.sidebar.header("Selectare perioada")
interval_date = st.sidebar.date_input("Perioada analizata", value=[date["data"].min().date(), date["data"].max().date()], min_value=date["data"].min().date(), max_value=date["data"].max().date())

if len(interval_date) == 2:
    data_start = pd.to_datetime(interval_date[0])
    data_final = pd.to_datetime(interval_date[1])
else:
    data_start = date["data"].min()
    data_final = date["data"].max()

date["probabilitate_predictie"] = model.predict_proba(date[COLOANE_TRASATURI])[:, 1]
date["alerta_precipitatii"] = date["cant_precipitatii"] >= prag_precipitatii
date["alerta_probabilitate"] = date["probabilitate_predictie"] >= prag_probabilitate
date["alerta_scor_risc"] = date["scor_risc"] >= prag_scor_risc
date["culoare_precipitatii"] = date["alerta_precipitatii"].map({True: "Peste prag", False: "Normal"})

date_filtrate = date[(date["data"] >= data_start) & (date["data"] <= data_final)].copy()
if date_filtrate.empty:
    st.warning("Nu exista date pentru perioada selectata.")
    st.stop()

rand_selectat = date_filtrate.iloc[-1].copy()

# ==============================
# SIMULARE CURENTA SI STATUS ALERTA
# ==============================

st.sidebar.header("Actualizare variabile")
temperatura_minima = st.sidebar.number_input("Temperatura minima", value=float(rand_selectat["temp_aer_min"]), step=0.1)
temperatura_maxima = st.sidebar.number_input("Temperatura maxima", value=float(rand_selectat["temp_aer_max"]), step=0.1)
viteza_maxima_vant = st.sidebar.number_input("Viteza maxima vant", value=float(rand_selectat["viteza_vant_max"]), step=0.1)
precipitatii = st.sidebar.number_input("Precipitatii", value=float(rand_selectat["cant_precipitatii"]), step=0.1)

situatie_curenta = rand_selectat.copy()
situatie_curenta["temp_aer_min"] = temperatura_minima
situatie_curenta["temp_aer_max"] = temperatura_maxima
situatie_curenta["viteza_vant_max"] = viteza_maxima_vant
situatie_curenta["cant_precipitatii"] = precipitatii
situatie_curenta["variatie_temp"] = temperatura_maxima - temperatura_minima

if "zile_ploioase_consecutive" not in situatie_curenta.index:
    situatie_curenta["zile_ploioase_consecutive"] = 0

situatie_curenta["scor_risc"] = (
    2 * int(precipitatii >= prag_precipitatii) +
    int(viteza_maxima_vant >= date["viteza_vant_max"].quantile(0.90)) +
    int(temperatura_maxima >= date["temp_aer_max"].quantile(0.95)) +
    int(temperatura_minima <= date["temp_aer_min"].quantile(0.05)) +
    int(situatie_curenta["zile_ploioase_consecutive"] >= 3)
)

esantion_curent = pd.DataFrame([situatie_curenta[COLOANE_TRASATURI]])
probabilitate = model.predict_proba(esantion_curent)[0][1]

if precipitatii >= prag_precipitatii or probabilitate >= prag_probabilitate or situatie_curenta["scor_risc"] >= prag_scor_risc:
    status = "ALERTA ROSIE"
    culoare_status = "red"
elif probabilitate >= 0.60 or situatie_curenta["scor_risc"] == prag_scor_risc - 1:
    status = "ALERTA PORTOCALIE"
    culoare_status = "orange"
elif probabilitate >= 0.40:
    status = "ALERTA GALBENA"
    culoare_status = "gold"
else:
    status = "NORMAL"
    culoare_status = "green"

if status == "ALERTA ROSIE":
    st.error("ALERTA ROSIE - Pragul hibrid de risc a fost depasit.")
    if trebuie_trimisa_alerta(status):
        trimite_alerta_email(probabilitate, precipitatii, status, situatie_curenta.get("scor_risc"), situatie_curenta.get("temp_aer_max"), situatie_curenta.get("viteza_vant_max"), situatie_curenta.get("zile_ploioase_consecutive"))
elif status == "ALERTA PORTOCALIE":
    st.warning("ALERTA PORTOCALIE - Risc climatic hibrid ridicat.")
elif status == "ALERTA GALBENA":
    st.warning("ALERTA GALBENA - Risc climatic moderat.")
else:
    st.success("CONDITII NORMALE")

# ==============================
# INDICATORI SI GRAFICE
# ==============================

coloana_kpi1, coloana_kpi2, coloana_kpi3, coloana_kpi4, coloana_kpi5, coloana_kpi6 = st.columns(6)
with coloana_kpi1: st.metric("Risc ML", f"{probabilitate * 100:.1f}%")
with coloana_kpi2: st.metric("Precipitatii", f"{precipitatii:.1f} mm")
with coloana_kpi3: st.metric("Vant", f"{viteza_maxima_vant:.1f}")
with coloana_kpi4: st.metric("Scor risc", int(situatie_curenta["scor_risc"]))
with coloana_kpi5: st.metric("Evenimente extreme", int(date_filtrate["eveniment_extrem"].sum()))
with coloana_kpi6: st.metric("Status", status)

tabel_tendinte = date_filtrate.set_index("data").resample("M").agg({"cant_precipitatii": "sum", "probabilitate_predictie": "mean", "scor_risc": "mean", "eveniment_extrem": "sum"}).reset_index()
inaltime_grafic = 200
margini_grafic = dict(l=20, r=10, t=45, b=25)
font_grafic = dict(size=10)
marime_titlu_grafic = 14

rand1_col1, rand1_col2, rand1_col3 = st.columns(3)
with rand1_col1:
    fig = go.Figure(go.Indicator(mode="gauge", value=probabilitate * 100, title={"text": "Risc ML (%)", "font": {"size": marime_titlu_grafic}}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": culoare_status}, "steps": [{"range": [0, 40], "color": "lightgreen"}, {"range": [40, 60], "color": "yellow"}, {"range": [60, 80], "color": "orange"}, {"range": [80, 100], "color": "red"}]}))
    fig.update_layout(height=inaltime_grafic, margin=margini_grafic, font=font_grafic)
    st.plotly_chart(fig, use_container_width=True)
with rand1_col2:
    fig = go.Figure(go.Indicator(mode="gauge", value=int(situatie_curenta["scor_risc"]), title={"text": "Scor hibrid", "font": {"size": marime_titlu_grafic}}, gauge={"axis": {"range": [0, 8]}, "bar": {"color": culoare_status}, "steps": [{"range": [0, 1], "color": "lightgreen"}, {"range": [1, 2], "color": "yellow"}, {"range": [2, 4], "color": "orange"}, {"range": [4, 8], "color": "red"}]}))
    fig.update_layout(height=inaltime_grafic, margin=margini_grafic, font=font_grafic)
    st.plotly_chart(fig, use_container_width=True)
with rand1_col3:
    fig = px.histogram(date_filtrate, x="probabilitate_predictie", nbins=16, color="eveniment_extrem", title="Distributia probabilitatilor", height=inaltime_grafic)
    fig.add_vline(x=0.30, line_dash="dash", line_color="red")
    fig.update_layout(margin=margini_grafic, title_font_size=marime_titlu_grafic, font=font_grafic, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

rand2_col1, rand2_col2, rand2_col3 = st.columns(3)
with rand2_col1:
    evolutie_lunara = date_filtrate.set_index("data").resample("M").agg({"probabilitate_predictie": "mean", "scor_risc": "mean"}).reset_index()
    fig = px.line(evolutie_lunara, x="data", y="probabilitate_predictie", title="Evolutia lunara a riscului estimat", height=inaltime_grafic)
    fig.add_hline(y=prag_probabilitate, line_dash="dash", line_color="red")
    fig.update_traces(line=dict(width=2))
    fig.update_layout(margin=margini_grafic, title_font_size=marime_titlu_grafic, font=font_grafic, showlegend=False, xaxis_title="Luna", yaxis_title="Prob. medie ML")
    st.plotly_chart(fig, use_container_width=True)
with rand2_col2:
    fig = px.bar(date_filtrate, x="data", y="cant_precipitatii", color="culoare_precipitatii", color_discrete_map={"Normal": "#4C78A8", "Peste prag": "red"}, title="Evolutia precipitatiilor", height=inaltime_grafic)
    fig.add_hline(y=prag_precipitatii, line_dash="dash", line_color="red")
    fig.update_layout(margin=margini_grafic, title_font_size=marime_titlu_grafic, font=font_grafic, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with rand2_col3:
    fig = px.line(date_filtrate, x="data", y="scor_risc", title="Evolutia scorului hibrid de risc", height=inaltime_grafic)
    fig.add_hline(y=prag_scor_risc, line_dash="dash", line_color="red")
    fig.update_layout(margin=margini_grafic, title_font_size=marime_titlu_grafic, font=font_grafic)
    st.plotly_chart(fig, use_container_width=True)

rand3_col1, rand3_col2, rand3_col3 = st.columns(3)
with rand3_col1:
    fig = px.box(date_filtrate, x="eveniment_extrem", y="cant_precipitatii", color="eveniment_extrem", title="Precipitatii: normal vs extrem", height=inaltime_grafic)
    fig.update_layout(margin=margini_grafic, title_font_size=marime_titlu_grafic, font=font_grafic, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with rand3_col2:
    date_temperatura = date_filtrate[(date_filtrate["temp_aer_max"] > -50) & (date_filtrate["temp_aer_max"] < 60)].copy()
    fig = px.box(date_temperatura, x="eveniment_extrem", y="temp_aer_max", color="eveniment_extrem", title="Temperatura maxima: normal vs extrem", height=inaltime_grafic)
    fig.update_layout(margin=margini_grafic, title_font_size=marime_titlu_grafic, font=font_grafic, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with rand3_col3:
    fig = px.line(tabel_tendinte, x="data", y=["cant_precipitatii", "eveniment_extrem"], title="Tendinta lunara a precipitatiilor si a evenimentelor extreme", height=inaltime_grafic)
    fig.update_layout(margin=margini_grafic, title_font_size=marime_titlu_grafic, font=font_grafic, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
