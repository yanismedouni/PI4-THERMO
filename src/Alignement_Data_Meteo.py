import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------------
# CHEMINS — À MODIFIER UNIQUEMENT ICI 
# pour copier un chemi d'accès à insérer, voir image jointe au répertoire
# le path out_path donnez lui le nom que vous souhaitez, un fichier sera créé.
# ------------------------------------------------------------------
in_path = r"C:\Users\Samia\OneDrive - polymtlus\Bureau\DataThermo\california\15minute_data_california.csv"
temp_path = r"C:\Users\Samia\OneDrive - polymtlus\Bureau\DataThermo\california\meteo_californie.csv"
out_path = r"C:\Users\Samia\OneDrive - polymtlus\Bureau\DataThermo\california\californie_matched.csv"


# ------------------------------------------------------------------
# Helper: lecture robuste Open-Meteo (saute les métadonnées)
# ------------------------------------------------------------------
def read_open_meteo_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("time,"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Entête 'time,' introuvable dans le fichier météo.")

    from io import StringIO
    csv_txt = "".join(lines[header_idx:])
    df = pd.read_csv(StringIO(csv_txt))

    # Parse en ignorant toute timezone (on force UTC puis on enlève le tz)
    dt_utc = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["time"] = dt_utc.dt.tz_localize(None)

    df = df.dropna(subset=["time"]).sort_values("time")
    return df


# ------------------------------------------------------------------
# 1) Lecture du fichier principal (pas 15 min)
# ------------------------------------------------------------------
data = pd.read_csv(in_path)

if "local_15min" not in data.columns:
    raise ValueError("Colonne 'local_15min' introuvable dans le fichier principal.")


# ------------------------------------------------------------------
# 1bis) Nettoyage timestamp (première étape)
# ------------------------------------------------------------------
df = data.copy()

s = df["local_15min"].astype(str).str.strip()
s = s.str.replace("T", " ", regex=False)
s = s.str.replace(r"\.\d{1,6}", "", regex=True)
s = s.str.replace(r"[+-]\d{2}:?\d{2}$", "", regex=True).str.strip()

dt_utc = pd.to_datetime(s, errors="coerce", utc=True)
df["local_15min"] = dt_utc.dt.tz_localize(None)

print("dtype après parse:", df["local_15min"].dtype)  # doit afficher datetime64[ns]
print("NaT:", df["local_15min"].isna().sum())

df = df.dropna(subset=["local_15min"]).sort_values("local_15min")

df["year"]   = df["local_15min"].dt.year
df["month"]  = df["local_15min"].dt.month
df["day"]    = df["local_15min"].dt.day
df["hour"]   = df["local_15min"].dt.hour
df["minute"] = df["local_15min"].dt.minute


# ------------------------------------------------------------------
# 2) Lecture du fichier météo et extraction température
# ------------------------------------------------------------------
meteo = read_open_meteo_csv(temp_path)

temp_candidates = [c for c in meteo.columns if "temperature" in c.lower()]
if len(temp_candidates) == 0:
    raise ValueError("Aucune colonne contenant 'temperature' trouvée.")
temp_col = temp_candidates[0]

meteo = meteo[["time", temp_col]].rename(columns={"time": "meteo_time", temp_col: "temp_out"})


# ------------------------------------------------------------------
# 3) Alignement temporel à l'heure
# ------------------------------------------------------------------
df["hour_time"] = df["local_15min"].dt.floor("h")

meteo["hour_time"] = meteo["meteo_time"].dt.floor("h")
meteo = meteo.drop_duplicates(subset=["hour_time"]).sort_values("hour_time")

df = df.merge(meteo[["hour_time", "temp_out"]], on="hour_time", how="left")


# ------------------------------------------------------------------
# 4) Écriture du fichier fusionné
# ------------------------------------------------------------------
os.makedirs(os.path.dirname(out_path), exist_ok=True)

df.drop(columns=["hour_time"], inplace=True)
df.to_csv(out_path, index=False)

print("Fichier généré :", out_path)
print("Nombre de lignes :", len(df))
print("Valeurs manquantes température :", int(df["temp_out"].isna().sum()))
