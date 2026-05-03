"""
ENM412 – MAN Türkiye A.Ş. Stok Yönetimi Modernizasyonu
Modül 2 – 4 ML Modeli: RF · XGBoost · CatBoost · LightGBM + Optuna
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN

Train: Ay 1-30  |  Test: Ay 31-36 (son 6 ay)
Karşılaştırma: ML modelleri + Geleneksel yöntemler (HO, Üstel, Naif)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from m1_veri import FEATURE_COLS, TARGET_COL, PARCA_COL, geleneksel_tahmin

# ── Opsiyonel kütüphaneler ───────────────────────────────────────
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("[!] xgboost bulunamadı → GradientBoosting kullanılacak")

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False
    print("[!] lightgbm bulunamadı → GradientBoosting kullanılacak")

try:
    from catboost import CatBoostRegressor
    CAT_OK = True
except ImportError:
    CAT_OK = False
    print("[!] catboost bulunamadı → GradientBoosting kullanılacak")


# ══════════════════════════════════════════════════════════════════
# METRİKLER
# ══════════════════════════════════════════════════════════════════

def mae(y, yhat):
    return float(mean_absolute_error(y, yhat))

def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))

def mape(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    mask = y > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)

def metrik_hesapla(y_true, y_pred, model_adi=""):
    return {
        "model":  model_adi,
        "MAE":    mae(y_true, y_pred),
        "RMSE":   rmse(y_true, y_pred),
        "MAPE":   mape(y_true, y_pred),
    }


# ══════════════════════════════════════════════════════════════════
# OPTUNA OBJEKTİFLERİ
# ══════════════════════════════════════════════════════════════════

def _get_X_y(df):
    feat = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat].fillna(0).values
    y = df[TARGET_COL].values
    return X, y, feat


def _rf_objective(trial, X, y):
    p = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
        "max_depth":         trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_float("max_features", 0.3, 1.0),
        "random_state": 42, "n_jobs": -1,
    }
    # 5-fold zaman serisi CV
    fold_size = len(X) // 6
    scores = []
    for fold in range(1, 6):
        tr_end = fold * fold_size
        if tr_end >= len(X):
            break
        m = RandomForestRegressor(**p)
        m.fit(X[:tr_end], y[:tr_end])
        scores.append(rmse(y[tr_end:tr_end+fold_size], m.predict(X[tr_end:tr_end+fold_size])))
    return np.mean(scores) if scores else 1e9


def _xgb_objective(trial, X, y):
    if XGB_OK:
        p = {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 400),
            "max_depth":       trial.suggest_int("max_depth", 2, 8),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42, "n_jobs": -1, "verbosity": 0,
        }
        ModelClass = xgb.XGBRegressor
    else:
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
            "max_depth":         trial.suggest_int("max_depth", 2, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "random_state": 42,
        }
        ModelClass = GradientBoostingRegressor

    fold_size = len(X) // 6
    scores = []
    for fold in range(1, 6):
        tr_end = fold * fold_size
        if tr_end >= len(X):
            break
        m = ModelClass(**p)
        m.fit(X[:tr_end], y[:tr_end])
        scores.append(rmse(y[tr_end:tr_end+fold_size], m.predict(X[tr_end:tr_end+fold_size])))
    return np.mean(scores) if scores else 1e9


def _lgb_objective(trial, X, y):
    if LGB_OK:
        p = {
            "n_estimators":   trial.suggest_int("n_estimators", 50, 400),
            "max_depth":      trial.suggest_int("max_depth", 2, 10),
            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":     trial.suggest_int("num_leaves", 15, 127),
            "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":      trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        ModelClass = lgb.LGBMRegressor
    else:
        p = {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
            "max_depth":     trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "random_state": 42,
        }
        ModelClass = GradientBoostingRegressor

    fold_size = len(X) // 6
    scores = []
    for fold in range(1, 6):
        tr_end = fold * fold_size
        if tr_end >= len(X):
            break
        m = ModelClass(**p)
        m.fit(X[:tr_end], y[:tr_end])
        scores.append(rmse(y[tr_end:tr_end+fold_size], m.predict(X[tr_end:tr_end+fold_size])))
    return np.mean(scores) if scores else 1e9


def _cat_objective(trial, X, y):
    if CAT_OK:
        p = {
            "iterations":      trial.suggest_int("iterations", 50, 400),
            "depth":           trial.suggest_int("depth", 2, 8),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg":     trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
            "random_seed": 42, "verbose": 0,
        }
        ModelClass = CatBoostRegressor
    else:
        p = {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
            "max_depth":     trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "random_state": 42,
        }
        ModelClass = GradientBoostingRegressor

    fold_size = len(X) // 6
    scores = []
    for fold in range(1, 6):
        tr_end = fold * fold_size
        if tr_end >= len(X):
            break
        m = ModelClass(**p)
        m.fit(X[:tr_end], y[:tr_end])
        scores.append(rmse(y[tr_end:tr_end+fold_size], m.predict(X[tr_end:tr_end+fold_size])))
    return np.mean(scores) if scores else 1e9


# ══════════════════════════════════════════════════════════════════
# KÜME BAZLI MODEL EĞİTİMİ
# ══════════════════════════════════════════════════════════════════

def kume_modelleri_egit(train_df: pd.DataFrame,
                        test_df:  pd.DataFrame,
                        segment_col: str = "Segment",
                        n_trials: int = 30) -> dict:
    """
    Her segment (AX, AY, BX, BY...) için 4 model eğitir.
    Test seti üzerinde MAE/RMSE/MAPE hesaplar.
    Şampiyon modeli seçer (en düşük RMSE).

    Returns
    -------
    dict: {segment: {model_adi: model, metrikler, sampiyon, ...}}
    """
    segmentler = train_df[segment_col].dropna().unique() if segment_col in train_df.columns \
                 else ["ALL"]

    sonuclar = {}

    for seg in segmentler:
        if segment_col in train_df.columns:
            tr = train_df[train_df[segment_col] == seg]
            te = test_df[test_df[segment_col]   == seg]
        else:
            tr, te = train_df, test_df

        if len(tr) < 20:
            continue

        X_tr, y_tr, feat = _get_X_y(tr)
        X_te, y_te, _    = _get_X_y(te)

        modeller   = {}
        tahminler  = {}
        metrikler  = {}

        # ── RF ─────────────────────────────────────────────────
        print(f"  [{seg}] RF...")
        s = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(lambda t: _rf_objective(t, X_tr, y_tr), n_trials=n_trials,
                   show_progress_bar=False)
        p = {**s.best_params, "random_state": 42, "n_jobs": -1}
        m = RandomForestRegressor(**p).fit(X_tr, y_tr)
        pred = np.maximum(m.predict(X_te), 0)
        modeller["rf"]   = m
        tahminler["rf"]  = pred
        metrikler["rf"]  = metrik_hesapla(y_te, pred, "RF")

        # ── XGBoost ────────────────────────────────────────────
        print(f"  [{seg}] XGBoost...")
        s = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(lambda t: _xgb_objective(t, X_tr, y_tr), n_trials=n_trials,
                   show_progress_bar=False)
        if XGB_OK:
            p = {**s.best_params, "random_state": 42, "n_jobs": -1, "verbosity": 0}
            m = xgb.XGBRegressor(**p).fit(X_tr, y_tr)
        else:
            p = {**s.best_params, "random_state": 42}
            m = GradientBoostingRegressor(**p).fit(X_tr, y_tr)
        pred = np.maximum(m.predict(X_te), 0)
        modeller["xgb"]  = m
        tahminler["xgb"] = pred
        metrikler["xgb"] = metrik_hesapla(y_te, pred, "XGBoost")

        # ── LightGBM ───────────────────────────────────────────
        print(f"  [{seg}] LightGBM...")
        s = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(lambda t: _lgb_objective(t, X_tr, y_tr), n_trials=n_trials,
                   show_progress_bar=False)
        if LGB_OK:
            p = {**s.best_params, "random_state": 42, "n_jobs": -1, "verbose": -1}
            m = lgb.LGBMRegressor(**p).fit(X_tr, y_tr)
        else:
            p = {**s.best_params, "random_state": 42}
            m = GradientBoostingRegressor(**p).fit(X_tr, y_tr)
        pred = np.maximum(m.predict(X_te), 0)
        modeller["lgb"]  = m
        tahminler["lgb"] = pred
        metrikler["lgb"] = metrik_hesapla(y_te, pred, "LightGBM")

        # ── CatBoost ───────────────────────────────────────────
        print(f"  [{seg}] CatBoost...")
        s = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(lambda t: _cat_objective(t, X_tr, y_tr), n_trials=n_trials,
                   show_progress_bar=False)
        if CAT_OK:
            p = {**s.best_params, "random_seed": 42, "verbose": 0}
            m = CatBoostRegressor(**p).fit(X_tr, y_tr)
        else:
            p_gb = {k: v for k, v in s.best_params.items()
                    if k in ["n_estimators","max_depth","learning_rate"]}
            p_gb["random_state"] = 42
            m = GradientBoostingRegressor(**p_gb).fit(X_tr, y_tr)
        pred = np.maximum(m.predict(X_te), 0)
        modeller["cat"]  = m
        tahminler["cat"] = pred
        metrikler["cat"] = metrik_hesapla(y_te, pred, "CatBoost")

        # ── Şampiyon Seçimi ────────────────────────────────────
        sampiyon = min(metrikler, key=lambda k: metrikler[k]["RMSE"])
        rmse_str = " | ".join([f"{k.upper()}={metrikler[k]['RMSE']:,.0f}"
                                for k in ["rf","xgb","lgb","cat"]])
        print(f"  [{seg}] {rmse_str} → Şampiyon: {sampiyon.upper()}")

        # Feature importance (şampiyon modelden)
        samp_model = modeller[sampiyon]
        try:
            fi = getattr(samp_model, "feature_importances_", None)
        except:
            fi = None

        sonuclar[seg] = {
            "modeller":    modeller,
            "tahminler":   tahminler,
            "metrikler":   metrikler,
            "sampiyon":    sampiyon,
            "feat_cols":   feat,
            "y_test":      y_te,
            "X_test":      X_te,
            "feature_importance": fi,
        }

    return sonuclar


# ══════════════════════════════════════════════════════════════════
# PARÇA BAZLI TAHMİN
# ══════════════════════════════════════════════════════════════════

def parca_tahmin(parca_kodu: str,
                 ml_df: pd.DataFrame,
                 segment_modelleri: dict,
                 n_ay: int = 6) -> dict:
    """
    Tek parça için:
    1. Train verisiyle model seç (segment bazlı şampiyon)
    2. Test (son 6 ay) üzerinde tahmin yap → MAE/RMSE/MAPE
    3. Geleneksel yöntemlerle karşılaştır
    4. Gelecek n_ay tahmini üret (test verisi son değerinden devam)

    Returns
    -------
    dict: tahminler, metrikler_ml, metrikler_geleneksel, sampiyon, y_test, y_pred_test
    """
    from m1_veri import parca_verisi, FEATURE_COLS

    pv = parca_verisi(ml_df, parca_kodu)
    ts_train = pv["ts_train"]
    ts_test  = pv["ts_test"]
    train_df = pv["train"]
    test_df  = pv["test"]

    # Segment
    seg = train_df["Segment"].iloc[0] if "Segment" in train_df.columns else "AY"
    if seg not in segment_modelleri:
        seg = list(segment_modelleri.keys())[0]

    seg_res   = segment_modelleri[seg]
    sampiyon  = seg_res["sampiyon"]
    model     = seg_res["modeller"][sampiyon]
    feat_cols = seg_res["feat_cols"]

    # Test tahmini (parça bazlı)
    X_te = test_df[[c for c in feat_cols if c in test_df.columns]].fillna(0).values
    if X_te.shape[1] < len(feat_cols):
        # eksik sütunları 0 ile doldur
        full = np.zeros((len(test_df), len(feat_cols)))
        for i, c in enumerate(feat_cols):
            if c in test_df.columns:
                full[:, i] = test_df[c].fillna(0).values
        X_te = full

    y_pred_test = np.maximum(model.predict(X_te), 0)
    y_true_test = ts_test

    ml_metrikler = {
        sampiyon.upper(): metrik_hesapla(y_true_test, y_pred_test, sampiyon.upper())
    }
    # Diğer modellerin de test tahminlerini al
    for k, m in seg_res["modeller"].items():
        if k != sampiyon:
            pred_k = np.maximum(m.predict(X_te), 0)
            ml_metrikler[k.upper()] = metrik_hesapla(y_true_test, pred_k, k.upper())

    # Geleneksel yöntemler
    gel = geleneksel_tahmin(ts_train, n_tahmin=len(ts_test))
    gel_metrikler = {
        "Hareketli Ort.": metrik_hesapla(y_true_test, gel["hareketli_ort"], "Hareketli Ort."),
        "Üstel Düzeltme": metrik_hesapla(y_true_test, gel["ustel"],         "Üstel Düzeltme"),
        "Naif":           metrik_hesapla(y_true_test, gel["naif"],           "Naif"),
    }

    # Gelecek n_ay tahmini
    # Son test satırını başlangıç noktası olarak al, sonraki ayları iteratif tahmin et
    son_satir = test_df.iloc[-1].copy()
    tahminler = []
    for i in range(n_ay):
        x = np.array([[son_satir.get(c, 0.0) if pd.notna(son_satir.get(c, 0.0)) else 0.0
                        for c in feat_cols]])
        pred = max(float(model.predict(x)[0]), 0.0)
        tahminler.append(pred)
        # lag özelliklerini güncelle (basit)
        if "lag_1" in son_satir.index:
            son_satir["lag_3"] = son_satir.get("lag_1", 0)
            son_satir["lag_1"] = pred

    return {
        "tahminler":       tahminler,
        "y_test":          y_true_test.tolist(),
        "y_pred_test":     y_pred_test.tolist(),
        "ts_train":        ts_train.tolist(),
        "sampiyon":        sampiyon.upper(),
        "segment":         seg,
        "ml_metrikler":    ml_metrikler,
        "gel_metrikler":   gel_metrikler,
        "tum_ml_pred":     {k.upper(): np.maximum(seg_res["modeller"][k].predict(X_te), 0).tolist()
                             for k in seg_res["modeller"]},
    }


# ══════════════════════════════════════════════════════════════════
# BATCH TAHMİN (tüm parçalar)
# ══════════════════════════════════════════════════════════════════

def batch_tahmin(ml_df: pd.DataFrame,
                 segment_modelleri: dict,
                 parcalar: list,
                 n_ay: int = 6) -> pd.DataFrame:
    rows = []
    for pid in parcalar:
        try:
            res = parca_tahmin(pid, ml_df, segment_modelleri, n_ay)
            rec = {"Parça_Kodu": pid, "Sampiyon": res["sampiyon"], "Segment": res["segment"]}
            for i, t in enumerate(res["tahminler"], 1):
                rec[f"Tahmin_Ay_{i}"] = round(t, 1)
            samp_met = res["ml_metrikler"].get(res["sampiyon"], {})
            rec["MAE"]  = round(samp_met.get("MAE",  0), 2)
            rec["RMSE"] = round(samp_met.get("RMSE", 0), 2)
            rec["MAPE"] = round(samp_met.get("MAPE", 0), 2) if not np.isnan(samp_met.get("MAPE", np.nan)) else None
            rows.append(rec)
        except Exception as e:
            print(f"  [!] {pid}: {e}")
    return pd.DataFrame(rows)
