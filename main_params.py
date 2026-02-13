"""
Code pour importer les données et les prétraiter.

Created on Fri Feb 12 2026
@author: catherinehenri
"""
from __future__ import annotations

import sys
from pathlib import Path

import meteostat as ms

from pretraitement import (
    load_data,
    parse_naive_datetime_col,
    clip_negative_cols,
    compute_equipment_series_sum,
    filter_by_ids,
)
from weather import RegionSpec
from probability import (
    EquipmentSpec,
    ProbaRunConfig,
    attach_region_column,
    estimate_proba_on_by_temp_multi_region,
)
from plotting import plot_proba_curve


def _home_path(*parts: str) -> str:
    return str(Path.home().joinpath(*parts))


SITE_CONFIG = {
    "newyork": {
        "csv_path": _home_path(
            "Desktop", "Genie Elec", "Session H2026", "ELE8080", "Dev",
            "15minute_data_newyork", "15minute_data_newyork.csv"
        ),
        "usecols": ["dataid", "local_15min", "air1", "furnace1", "furnace2"],
        "id_to_region": {
            27: "Brooktondale",
            558: "Ithaca",
            950: "Ithaca",
            1240: "Groton",
            1417: "Ithaca",
            3000: "Ithaca",
            3488: "Ithaca",
            3517: "Ithaca",
            5058: "Ithaca",
            5587: "Ithaca",
        },
        "regions": {
            "Brooktondale": RegionSpec("Brooktondale", ms.Point(42.38056, -76.39472), "America/New_York"),
            "Ithaca":       RegionSpec("Ithaca",       ms.Point(42.443962, -76.501884), "America/New_York"),
            "Groton":       RegionSpec("Groton",       ms.Point(42.589639, -76.367194), "America/New_York"),
        },
        "equipment": {
            "heat": dict(
                name="Chauffage",
                energy_col="furnace_total",
                seuil_on=0.1,
                months=(5, 10),
                bin_start=-10,
                bin_stop=30,
                bin_step=1,
                use_ge=True,
            ),
            "ac": dict(
                name="Climatisation",
                energy_col="air1",
                seuil_on=0.1,
                months=(6, 7, 8, 9),
                bin_start=15,
                bin_stop=45,
                bin_step=1,
                use_ge=True,
            ),
        },
    },

    "austin": {
        "csv_path": _home_path(
            "Desktop", "Genie Elec", "Session A2025", "ELE8080", "Dev",
            "15minute_data_austin", "15minute_data_austin.csv"
        ),
        "usecols": ["dataid", "local_15min", "air1"],
        "id_to_region": {
            661: "Austin",
            1642: "Austin",
            2335: "Austin",
            2361: "Austin",
            2818: "Austin",
            3039: "Austin",
            3456: "Austin",
            3538: "Austin",
            4031: "Austin",
            4373: "Austin",
            4767: "Austin",
            5746: "Austin",
        },
        "regions": {
            "Austin": RegionSpec("Austin", ms.Point(30.2672, -97.7431, 150), "America/Chicago"),
        },
        "equipment": {
            "ac": dict(
                name="Climatisation",
                energy_col="air1",
                seuil_on=0.1,
                months=(5, 6, 7, 8, 9, 10),
                bin_start=15,
                bin_stop=45,
                bin_step=1,
                use_ge=True,
            ),
        },
    },

    "california": {
        "csv_path": _home_path(
            "Desktop", "Genie Elec", "Session H2026", "ELE8080", "Dev",
            "15minute_data_california", "15minute_data_california.csv"
        ),
        "usecols": ["dataid", "local_15min", "air1"],
        "id_to_region": {
            203: "SanDiego",
            1450: "SanDiego",
            1524: "SanDiego",
            1731: "SanDiego",
            3938: "SanDiego",
            4495: "SanDiego",
            4934: "SanDiego",
            5938: "SanDiego",
            8061: "SanDiego",
            8342: "SanDiego",
            9775: "SanDiego",
        },
        "regions": {
            "SanDiego": RegionSpec("SanDiego", ms.Point(32.7157, -117.1611, 20), "America/Los_Angeles"),
        },
        "equipment": {
            "ac": dict(
                name="Climatisation",
                energy_col="air1",
                seuil_on=0.3,
                months=(6, 7, 8, 9),
                bin_start=15,
                bin_stop=45,
                bin_step=1,
                use_ge=True,
            ),
        },
    },
}


def _ensure_csv_exists(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV introuvable: {p}")


def _load_csv_robust(path: str, usecols: list[str]) -> "pd.DataFrame":
    """
    Charge un CSV en essayant plusieurs encodages (utile si le fichier n'est pas UTF-8).
    """
    import pandas as pd

    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, usecols=usecols, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Impossible de décoder le CSV (utf-8/cp1252/latin-1).")


def main(site: str = "newyork", equipment_keys: tuple[str, ...] = ("heat", "ac")) -> None:
    if site not in SITE_CONFIG:
        raise ValueError(f"Site inconnu: {site}. Choix: {list(SITE_CONFIG.keys())}")

    cfg = SITE_CONFIG[site]
    csv_path = cfg["csv_path"]
    _ensure_csv_exists(csv_path)

    usecols = cfg["usecols"]
    id_to_region = cfg["id_to_region"]
    regions = cfg["regions"]

    df = _load_csv_robust(csv_path, usecols=usecols)
    df = parse_naive_datetime_col(df, "local_15min")
    df = filter_by_ids(df, "dataid", sorted(set(id_to_region.keys())))

    energy_cols = [c for c in ("air1", "furnace1", "furnace2") if c in df.columns]
    df = clip_negative_cols(df, energy_cols)

    if "furnace1" in df.columns or "furnace2" in df.columns:
        df = compute_equipment_series_sum(df, ["furnace1", "furnace2"], out_col="furnace_total", min_count=1)
        df = clip_negative_cols(df, ["furnace_total"])

    df = attach_region_column(df, id_col="dataid", id_to_region=id_to_region, region_col="region", default_region=None)

    run_cfg = ProbaRunConfig(
        id_col="dataid",
        ts_col="local_15min",
        region_col="region",
        temp_col="temp",
        per_client=True,
        dropna_temp=True,
    )

    available_equipment = cfg["equipment"]

    for k in equipment_keys:
        if k not in available_equipment:
            continue

        equip = EquipmentSpec(**available_equipment[k])

        out = estimate_proba_on_by_temp_multi_region(
            df=df,
            regions=regions,
            equipment=equip,
            cfg=run_cfg,
            tz_source_meteostat="UTC",
            verbose=True,
        )

        ylabel = "P(ON)"
        if "clim" in equip.name.lower():
            ylabel = "P(clim ON)"
        elif "chauff" in equip.name.lower():
            ylabel = "P(chauffage ON)"

        plot_proba_curve(out, title=f"{equip.name} – {site}", ylabel=ylabel)


if __name__ == "__main__":
    site = (sys.argv[1].strip().lower() if len(sys.argv) >= 2 else "newyork")
    main(site=site)