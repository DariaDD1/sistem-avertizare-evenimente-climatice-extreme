# Sistem hibrid de avertizare timpurie a evenimentelor climatice extreme, folosind algoritmi de machine learning

Sistem hibrid de avertizare timpurie pentru evenimente climatice extreme, bazat pe date meteorologice istorice, scor hibrid de risc, algoritmi de Machine Learning, dashboard Streamlit si alerta prin email.

## Structura proiectului

```text
hybrid-climate-early-warning-system/
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── alerts.py
│   ├── dashboard.py
│   └── main.py
├── data/
├── outputs/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Instalare

```bash
pip install -r requirements.txt
```

## Rulare completa

```bash
python src/main.py --data-path data/Bucuresti-Baneasa_2006-2025.xlsx
```

## Pornire dashboard

```bash
streamlit run src/dashboard.py
```

## Rulare + pornire dashboard

```bash
python src/main.py --data-path data/Bucuresti-Baneasa_2006-2025.xlsx --run-dashboard
```

## Configurare alerta Gmail

Creeaza un fisier `.env` sau seteaza variabilele de mediu:

```bash
export GMAIL_SENDER="adresa_ta@gmail.com"
export GMAIL_APP_PASSWORD="parola_de_aplicatie"
export GMAIL_RECEIVER="destinatar@gmail.com"
```