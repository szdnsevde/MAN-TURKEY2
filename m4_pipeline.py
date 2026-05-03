"""
ENM412 – MAN Türkiye A.Ş.
Modül 4 – Ana Pipeline
Çalıştırma: python m4_pipeline.py --dosya MAN_ML_Dataset_v3.xlsx --n_trials 30
"""
import argparse, pickle, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from m1_veri    import veri_yukle
from m2_modeller import kume_modelleri_egit, batch_tahmin

def pipeline_calistir(dosya_yolu, n_trials=30, cache="enm412_cache.pkl"):
    t0 = time.time()
    print("="*65)
    print("  ENM412 – MAN TÜRKİYE A.Ş. STOK OPTİMİZASYON PİPELİNE")
    print("="*65)

    print("\n[1/4] Veri yükleniyor...")
    veri = veri_yukle(dosya_yolu)
    print(f"      {len(veri['parcalar']):,} parça | Train: {len(veri['train_df']):,} | Test: {len(veri['test_df']):,}")

    print(f"\n[2/4] 4 ML modeli eğitiliyor (RF·XGB·LGB·CAT) – {n_trials} trial/model...")
    seg_modelleri = kume_modelleri_egit(
        veri["train_df"], veri["test_df"],
        segment_col="Segment", n_trials=n_trials
    )

    # Segment bazlı özet
    print("\n  Segment Sonuçları:")
    for seg, res in seg_modelleri.items():
        m = res["metrikler"]
        rmse_str = " | ".join([f"{k.upper()}={m[k]['RMSE']:,.0f}" for k in m])
        print(f"  [{seg}] {rmse_str} → Şampiyon: {res['sampiyon'].upper()}")

    print("\n[3/4] Batch tahmin üretiliyor...")
    batch_df = batch_tahmin(veri["ml_df"], seg_modelleri, veri["parcalar"], n_ay=6)
    print(f"      {len(batch_df):,} parça tamamlandı.")

    sonuc = {
        "veri":           veri,
        "seg_modelleri":  seg_modelleri,
        "batch_df":       batch_df,
    }

    print(f"\n[4/4] Cache kaydediliyor → {cache}")
    with open(cache, "wb") as f:
        pickle.dump(sonuc, f, protocol=4)

    print(f"\n✅ Tamamlandı! Süre: {(time.time()-t0)/60:.1f} dk")
    print("="*65)
    return sonuc

def cache_yukle(cache="enm412_cache.pkl"):
    with open(cache,"rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dosya",    default="MAN_ML_Dataset_v3.xlsx")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--cache",    default="enm412_cache.pkl")
    args = p.parse_args()
    pipeline_calistir(args.dosya, args.n_trials, args.cache)
