"""
ENM412 – MAN Türkiye A.Ş. Stok Yönetimi Modernizasyonu
Modül 1 – Veri Yükleme ve Hazırlık
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN
Veri: MAN_ML_Dataset_v3.xlsx
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ML'e girecek özellik sütunları (Feature_Rehberi'nden - leakage riski olmayanlar)
FEATURE_COLS = [
    "lag_1", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6", "roll_std_3", "roll_std_6", "roll_max_3",
    "Ay", "Yil", "Ceyrek", "sin_ay", "cos_ay",
    "MPS_Toplam_Arac", "MPS_lag_1",
    "MPS_LC12m", "MPS_LC18m", "MPS_Coach", "MPS_Coach2", "MPS_Skyliner",
    "ABC_enc", "XYZ_enc",
]

TARGET_COL  = "Talep"
SPLIT_COL   = "Split"
PARCA_COL   = "Parça_Kodu"
TARIH_COL   = "Tarih"


def veri_yukle(dosya_yolu: str) -> dict:
    """
    Excel'deki tüm sayfaları okur, temiz dict döner.

    Returns
    -------
    dict içeriği:
        ml_df        : ML_Hazir_Veri (tüm satırlar)
        abc_df       : ABC/XYZ segmentasyon
        opt_df       : Optimizasyon parametreleri (maliyet vs.)
        mps_df       : MPS aylık üretim planı
        train_df     : Split=='Train' satırları
        test_df      : Split=='Test' satırları
        parcalar     : Parça kodu listesi
    """
    xl = pd.ExcelFile(dosya_yolu)

    # ── ML Hazır Veri ────────────────────────────────────────────
    ml = pd.read_excel(xl, sheet_name="ML_Hazir_Veri", header=0)
    ml[TARIH_COL] = ml[TARIH_COL].astype(str).str.strip()
    ml[PARCA_COL] = ml[PARCA_COL].astype(str).str.strip()

    # Özellik sütunlarındaki NaN'ları 0 ile doldur (lag başlangıç satırları)
    for c in FEATURE_COLS:
        if c in ml.columns:
            ml[c] = pd.to_numeric(ml[c], errors="coerce").fillna(0)

    ml[TARGET_COL] = pd.to_numeric(ml[TARGET_COL], errors="coerce").fillna(0)

    # ── ABC/XYZ Segmentasyon ─────────────────────────────────────
    abc = pd.read_excel(xl, sheet_name="ABC_XYZ_Segmentasyon", header=0)
    abc[PARCA_COL] = abc[PARCA_COL].astype(str).str.strip()

    # ── Optimizasyon Parametreleri ───────────────────────────────
    opt = pd.read_excel(xl, sheet_name="Optimizasyon_Parametreleri", header=0)
    opt[PARCA_COL] = opt[PARCA_COL].astype(str).str.strip()
    opt = opt.rename(columns={
        "Tedarik Süresi (gün)":    "LT_gun",
        "Birim Maliyet (TL)":      "Birim_Maliyet",
        "Sipariş Maliyeti (TL)":   "Siparis_Maliyeti",
        "Elde Tutma (TL/adet/ay)": "h",
        "Stoksuz Maliyet (TL)":    "p",
        "Başlangıç Stok":          "Baslangic_Stok",
    })
    opt["LT_ay"] = opt["LT_gun"] / 30

    # ── MPS ─────────────────────────────────────────────────────
    mps = pd.read_excel(xl, sheet_name="MPS_Long_Format", header=0)
    mps[TARIH_COL] = mps[TARIH_COL].astype(str).str.strip()

    # ── ABC/XYZ'yi ML verisine merge et ─────────────────────────
    ml = ml.merge(
        abc[[PARCA_COL, "Ort_Aylik_Talep", "Std_Sapma", "CV", "Segment"]],
        on=PARCA_COL, how="left", suffixes=("", "_abc")
    )
    # Optimizasyon parametrelerini de ekle
    ml = ml.merge(
        opt[[PARCA_COL, "LT_gun", "LT_ay", "Birim_Maliyet",
             "Siparis_Maliyeti", "h", "p", "Baslangic_Stok"]],
        on=PARCA_COL, how="left"
    )

    train_df = ml[ml[SPLIT_COL] == "Train"].copy().reset_index(drop=True)
    test_df  = ml[ml[SPLIT_COL] == "Test"].copy().reset_index(drop=True)
    parcalar = sorted(ml[PARCA_COL].unique().tolist())

    print(f"[Veri] {len(parcalar):,} parça | Train: {len(train_df):,} | Test: {len(test_df):,}")

    return {
        "ml_df":    ml,
        "abc_df":   abc,
        "opt_df":   opt,
        "mps_df":   mps,
        "train_df": train_df,
        "test_df":  test_df,
        "parcalar": parcalar,
    }


def parca_verisi(ml_df: pd.DataFrame, parca_kodu: str) -> dict:
    """Tek parça için train/test verisi ve zaman serisi döner."""
    pdf = ml_df[ml_df[PARCA_COL] == parca_kodu].sort_values(TARIH_COL)
    train = pdf[pdf[SPLIT_COL] == "Train"]
    test  = pdf[pdf[SPLIT_COL] == "Test"]

    ts_train = train[TARGET_COL].values
    ts_test  = test[TARGET_COL].values
    tarihler = pdf[TARIH_COL].tolist()

    return {
        "pdf":      pdf,
        "train":    train,
        "test":     test,
        "ts_train": ts_train,
        "ts_test":  ts_test,
        "tarihler": tarihler,
    }


def geleneksel_tahmin(ts_train: np.ndarray, n_tahmin: int = 6) -> dict:
    """
    Geleneksel yöntemlerle tahmin:
    - Hareketli Ortalama (son 6 ay)
    - Üstel Düzeltme (alpha=0.3)
    - Naif (son değer)

    Döner: {"hareketli_ort": [...], "ustel": [...], "naif": [...]}
    """
    n = len(ts_train)

    # Hareketli Ortalama (son 6 ay)
    pencere = min(6, n)
    ho_base = np.mean(ts_train[-pencere:])
    hareketli = [ho_base] * n_tahmin

    # Üstel Düzeltme (Basit)
    alpha = 0.3
    ustel_val = float(ts_train[0]) if n > 0 else 0.0
    for v in ts_train:
        ustel_val = alpha * float(v) + (1 - alpha) * ustel_val
    ustel = [ustel_val] * n_tahmin

    # Naif (son gözlem)
    naif_val = float(ts_train[-1]) if n > 0 else 0.0
    naif = [naif_val] * n_tahmin

    return {
        "hareketli_ort": hareketli,
        "ustel":         ustel,
        "naif":          naif,
    }
