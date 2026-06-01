import argparse
import subprocess
import sys
from pathlib import Path
from train_model import antreneaza_si_salveaza

# ==============================
# RULARE PIPELINE PRINCIPAL
# ==============================

def ruleaza_dashboard():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard.py"], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", nargs="+", required=True)
    parser.add_argument("--run-dashboard", action="store_true")
    argumente = parser.parse_args()

    for cale in argumente.data_path:
        if not Path(cale).exists():
            raise FileNotFoundError(f"Fisierul nu exista: {cale}")

    antreneaza_si_salveaza(argumente.data_path)
    print("\nPipeline-ul a fost finalizat.")
    print("Pentru dashboard ruleaza: streamlit run src/dashboard.py")

    if argumente.run_dashboard:
        ruleaza_dashboard()

if __name__ == "__main__":
    main()
