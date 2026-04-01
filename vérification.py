import pandas as pd

df = pd.read_csv("output/resultats_desagregation_3864_2015-07-02_7jours.csv")

# Pas de temps où clim ON mais grid faible
faux_on = df[(df["o_climatisation"] == 1) & (df["P_total"] < 0.5)]
print(f"Pas ON avec grid < 0.5 kW : {len(faux_on)}")
print(faux_on[["timestamp", "P_total", "T_ext", "P_estime_clim"]].head(10))