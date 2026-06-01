import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

FOLDER_REZULTATE = Path("outputs")
FOLDER_REZULTATE.mkdir(exist_ok=True)

# Trimitere email


def trimite_alerta_email(
    probabilitate,
    precipitatii,
    status,
    scor_risc=None,
    temperatura_maxima=None,
    viteza_vant=None,
    zile_ploioase_consecutive=None,
):
    email_expeditor = os.getenv("GMAIL_SENDER")
    parola_aplicatie = os.getenv("GMAIL_APP_PASSWORD")
    email_destinatar = os.getenv("GMAIL_RECEIVER")

    if not email_expeditor or not parola_aplicatie or not email_destinatar:
        print("Emailul nu a fost trimis. Lipsesc variabilele de mediu Gmail.")
        return False

    subiect = f"{status}: Avertizare eveniment climatic extrem"
    text_scor_risc = (
        f"Scor hibrid de risc: {scor_risc}\n"
        if scor_risc is not None
        else "Scor hibrid de risc: indisponibil\n"
    )
    text_temperatura = (
        f"Temperatura maxima: {temperatura_maxima:.2f} °C\n"
        if temperatura_maxima is not None
        else ""
    )
    text_vant = (
        f"Viteza maxima a vantului: {viteza_vant:.2f}\n"
        if viteza_vant is not None
        else ""
    )
    text_zile_ploioase = (
        f"Numar de zile ploioase consecutive: {zile_ploioase_consecutive}\n"
        if zile_ploioase_consecutive is not None
        else ""
    )

    corp_email = f"""
SISTEM HIBRID DE AVERTIZARE TIMPURIE PENTRU EVENIMENTE CLIMATICE EXTREME

Status alerta: {status}
Probabilitatea estimata a producerii unui eveniment extrem: {probabilitate * 100:.2f}%
{text_scor_risc}Ultima valoare a precipitatiilor: {precipitatii:.2f} mm
{text_temperatura}{text_vant}{text_zile_ploioase}
Interpretare:
Alerta a fost generata utilizand un sistem hibrid de decizie bazat pe probabilitatea ML, intensitatea precipitatiilor, persistenta precipitatiilor, conditiile de vant, temperaturile extreme si indicatorii temporali.

Actiune recomandata:
Verifica dashboardul si monitorizeaza evolutia conditiilor meteorologice.
"""

    mesaj = MIMEText(corp_email)
    mesaj["Subject"] = subiect
    mesaj["From"] = email_expeditor
    mesaj["To"] = email_destinatar

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(email_expeditor, parola_aplicatie)
            server.send_message(mesaj)
        print("Alerta Gmail a fost trimisa cu succes.")
        return True
    except smtplib.SMTPAuthenticationError:
        print("Autentificarea a esuat. Verifica parola de aplicatie Gmail.")
    except smtplib.SMTPException as eroare:
        print(f"A aparut o eroare SMTP: {eroare}")
    except Exception as eroare:
        print(f"A aparut o eroare neasteptata: {eroare}")
    return False


# Control trimitere alerta


def trebuie_trimisa_alerta(status):
    cale_jurnal = FOLDER_REZULTATE / "ultima_alerta_status.txt"
    if status != "ALERTA ROSIE":
        return False
    if not cale_jurnal.exists():
        cale_jurnal.write_text(f"{status}|{datetime.now()}", encoding="utf-8")
        return True
    status_anterior = cale_jurnal.read_text(encoding="utf-8").split("|")[0]
    if status_anterior != status:
        cale_jurnal.write_text(f"{status}|{datetime.now()}", encoding="utf-8")
        return True
    return False


# Testare directa a functiilor de alerta

if __name__ == "__main__":
    status_test = "ALERTA ROSIE"
    if trebuie_trimisa_alerta(status_test):
        trimite_alerta_email(0.91, 35.4, status_test, 3, 36.5, 18.2, 4)
    else:
        print("Alerta rosie a fost deja inregistrata.")
