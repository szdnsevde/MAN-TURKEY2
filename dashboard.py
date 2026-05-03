"""
ENM412 – MAN Türkiye A.Ş. Stok Yönetimi Modernizasyonu
Streamlit Dashboard
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN
Çalıştırma: streamlit run dashboard.py
"""
import sys, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="MAN Türkiye – Stok Optimizasyon",
                   page_icon="🏭", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.main{background:#F8F9FC}
.kart{background:white;border-radius:12px;padding:18px 14px;
      box-shadow:0 2px 10px rgba(0,0,0,.07);text-align:center;border-top:4px solid}
.kart-Q{border-color:#1E4D8C} .kart-r{border-color:#C8102E}
.kart-SS{border-color:#F39200} .kart-HZ{border-color:#00843D}
.kart-MAE{border-color:#7B2D8B} .kart-MAPE{border-color:#00838A}
.kart-RMSE{border-color:#5C5C8A}
.kart-baslik{font-size:11px;color:#666;font-weight:700;text-transform:uppercase;margin-bottom:5px}
.kart-deger{font-size:28px;font-weight:800;color:#1A1A2E;line-height:1.1}
.kart-alt{font-size:11px;color:#999;margin-top:3px}
.bolum{font-size:16px;font-weight:700;color:#1A1A2E;
        border-left:4px solid #1E4D8C;padding-left:10px;margin:20px 0 12px}
.aksiyon-yesil{background:linear-gradient(135deg,#00843D,#00A84F);border-radius:10px;
               padding:14px 22px;color:white;font-size:14px;font-weight:600;margin:12px 0}
.aksiyon-sari {background:linear-gradient(135deg,#F39200,#F5A623);border-radius:10px;
               padding:14px 22px;color:white;font-size:14px;font-weight:600;margin:12px 0}
.aksiyon-kirmizi{background:linear-gradient(135deg,#C8102E,#E01535);border-radius:10px;
                 padding:14px 22px;color:white;font-size:14px;font-weight:600;margin:12px 0}
.aksiyon-mavi{background:linear-gradient(135deg,#1E4D8C,#2D6FAE);border-radius:10px;
              padding:14px 22px;color:white;font-size:14px;font-weight:600;margin:12px 0}
</style>
""", unsafe_allow_html=True)

TARIH_TRAIN = [f"{y}-{m:02d}" for y in range(2022,2025) for m in range(1,13)][:30]
TARIH_TEST  = [f"{y}-{m:02d}" for y in range(2022,2025) for m in range(1,13)][30:36]
GELECEK     = ["2025-01","2025-02","2025-03","2025-04","2025-05","2025-06"]
ETIKET      = (["Oca-22","Şub-22","Mar-22","Nis-22","May-22","Haz-22",
                 "Tem-22","Ağu-22","Eyl-22","Eki-22","Kas-22","Ara-22",
                 "Oca-23","Şub-23","Mar-23","Nis-23","May-23","Haz-23",
                 "Tem-23","Ağu-23","Eyl-23","Eki-23","Kas-23","Ara-23",
                 "Oca-24","Şub-24","Mar-24","Nis-24","May-24","Haz-24"]
               +["Tem-24","Ağu-24","Eyl-24","Eki-24","Kas-24","Ara-24"]
               +["Oca-25","Şub-25","Mar-25","Nis-25","May-25","Haz-25"])

KART = '<div class="kart kart-{c}"><div class="kart-baslik">{b}</div><div class="kart-deger">{d}</div><div class="kart-alt">{a}</div></div>'

RENKLER = {
    "RF":"#1E4D8C","XGBoost":"#C8102E",
    "LightGBM":"#00843D","CatBoost":"#F39200",
    "Hareketli Ort.":"#888","Üstel Düzeltme":"#AAA","Naif":"#CCC",
}

# ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Model yükleniyor…")
def yukle(cache_yolu, veri_yolu):
    if Path(cache_yolu).exists():
        with open(cache_yolu,"rb") as f:
            return pickle.load(f)
    from m4_pipeline import pipeline_calistir
    return pipeline_calistir(veri_yolu, n_trials=20, cache=cache_yolu)

@st.cache_data(show_spinner=False)
def analiz_hesapla(_ml_df, _seg_mod, _opt_df, _abc_df, pid, grid_adim, n_trials):
    from m2_modeller import parca_tahmin
    from m3_optimizasyon import parca_optimize, aksiyon_uyarisi
    t_res = parca_tahmin(pid, _ml_df, _seg_mod, n_ay=6)
    o_res = parca_optimize(pid, _opt_df, _abc_df, t_res["tahminler"],
                           grid_adim=grid_adim, n_trials=n_trials)
    uyari = aksiyon_uyarisi(o_res, t_res["tahminler"])
    return t_res, o_res, uyari

# ──────────────────────────────────────────────────────────────────
# SİDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 MAN Türkiye A.Ş.")
    st.caption("ENM412 – Stok Yönetimi Dashboard")
    st.divider()
    st.subheader("⚙️ Sistem Ayarları")
    veri_yolu  = st.text_input("Veri Dosyası", value="MAN_ML_Dataset_v3.xlsx")
    cache_yolu = st.text_input("Model Cache",  value="enm412_cache.pkl")

    try:
        sistem = yukle(cache_yolu, veri_yolu)
        veri   = sistem["veri"]
        seg_mod= sistem["seg_modelleri"]
        batch  = sistem["batch_df"]
        st.success(f"✅ {len(veri['parcalar']):,} parça yüklendi")
    except Exception as e:
        st.error(f"Hata: {e}")
        st.stop()

    st.divider()
    st.subheader("🔍 Ürün Seçimi")
    pid = st.selectbox("Product ID", veri["parcalar"], index=0)

    st.divider()
    st.subheader("🎛️ Optimizasyon")
    grid_adim = st.slider("Grid Çözünürlüğü", 6, 20, 12)
    n_trials  = st.slider("Optuna Trial Sayısı", 20, 100, 40)
    hesapla   = st.button("🚀 Analizi Çalıştır", use_container_width=True, type="primary")
    st.divider()
    st.caption("Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN")
    st.caption("ENM412 – Endüstri Mühendisliğinde Tasarım II")

# ──────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#1A1A2E;font-size:26px;font-weight:800;'>🏭 Stok Yönetimi Optimizasyon Paneli</h1>", unsafe_allow_html=True)
st.caption("MAN Türkiye A.Ş. | ENM412 | RF · XGBoost · LightGBM · CatBoost + Grid Search + Optuna")
st.divider()

# Parça bilgisi
ml_df  = veri["ml_df"]
opt_df = veri["opt_df"]
abc_df = veri["abc_df"]

parca_abc = abc_df[abc_df["Parça_Kodu"]==pid]
seg_  = parca_abc["Segment"].values[0] if not parca_abc.empty else "—"
abc_  = parca_abc["ABC"].values[0]     if not parca_abc.empty else "—"
xyz_  = parca_abc["XYZ"].values[0]     if not parca_abc.empty else "—"
mu_   = float(parca_abc["Ort_Aylik_Talep"].values[0]) if not parca_abc.empty else 0

batch_row = batch[batch["Parça_Kodu"]==pid] if not batch.empty else pd.DataFrame()
sampiyon_ = batch_row["Sampiyon"].values[0] if not batch_row.empty else "—"

c1,c2,c3,c4,c5 = st.columns([3,1,1,1,2])
with c1: st.markdown(f"### 📦 {pid}")
with c2: st.markdown(f'<span style="background:#E8F0FB;color:#1E4D8C;padding:3px 10px;border-radius:16px;font-size:12px;font-weight:700;">ABC: {abc_}</span>', unsafe_allow_html=True)
with c3: st.markdown(f'<span style="background:#FFF3E0;color:#F39200;padding:3px 10px;border-radius:16px;font-size:12px;font-weight:700;">XYZ: {xyz_}</span>', unsafe_allow_html=True)
with c4: st.markdown(f'<span style="background:#F3E8FB;color:#7B2D8B;padding:3px 10px;border-radius:16px;font-size:12px;font-weight:700;">{seg_}</span>', unsafe_allow_html=True)
with c5: st.markdown(f"Ort. Talep (30 ay): **{mu_:,.0f}** adet/ay")

# ──────────────────────────────────────────────────────────────────
# ANALİZ
# ──────────────────────────────────────────────────────────────────
if hesapla or "analiz" not in st.session_state or st.session_state.get("son_pid")!=pid:
    with st.spinner(f"🔄 {pid} analiz ediliyor…"):
        try:
            t_res, o_res, uyari = analiz_hesapla(
                ml_df, seg_mod, opt_df, abc_df, pid, grid_adim, n_trials)
            st.session_state["analiz"]  = (t_res, o_res, uyari)
            st.session_state["son_pid"] = pid
        except Exception as e:
            st.error(f"Analiz hatası: {e}")
            st.stop()
else:
    t_res, o_res, uyari = st.session_state["analiz"]

# ──────────────────────────────────────────────────────────────────
# MODEL KARŞILAŞTIRMA METRİKLERİ
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="bolum">📊 Model Karşılaştırması – Test Seti (Son 6 Ay: Tem-24 → Ara-24)</div>', unsafe_allow_html=True)

ml_met  = t_res.get("ml_metrikler", {})
gel_met = t_res.get("gel_metrikler", {})
tum_met = {**ml_met, **gel_met}

if tum_met:
    met_rows = []
    for m_adi, m_val in tum_met.items():
        met_rows.append({
            "Model": m_adi,
            "MAE":   round(m_val.get("MAE",0),1),
            "RMSE":  round(m_val.get("RMSE",0),1),
            "MAPE":  f"{m_val.get('MAPE',0):.1f}%" if not pd.isna(m_val.get("MAPE",0)) else "—",
            "Tür":   "🤖 ML" if m_adi in ["RF","XGB","LGB","CAT","XGBOOST","LIGHTGBM","CATBOOST",
                                           t_res.get("sampiyon","")] else "📐 Geleneksel"
        })
    met_df = pd.DataFrame(met_rows)
    sampiyon_adi = t_res.get("sampiyon","")

    # Bar grafiği
    fig_met = go.Figure()
    for _, row in met_df.iterrows():
        renk = RENKLER.get(row["Model"], "#888")
        fig_met.add_trace(go.Bar(
            name=row["Model"], x=[row["Model"]], y=[row["RMSE"]],
            marker_color=renk,
            text=f"{row['RMSE']:,.0f}",
            textposition="outside",
        ))
    fig_met.update_layout(
        barmode="group", height=320,
        title="RMSE Karşılaştırması (düşük = iyi)",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="RMSE (adet)", gridcolor="#F0F0F0"),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(l=10,r=10,t=40,b=10),
    )
    st.plotly_chart(fig_met, use_container_width=True)

    # Tablo
    st.dataframe(met_df, hide_index=True, use_container_width=True,
                 column_config={"Model": st.column_config.TextColumn(width="medium"),
                                "Tür":   st.column_config.TextColumn(width="small")})

# ──────────────────────────────────────────────────────────────────
# TAHMİN GRAFİĞİ
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="bolum">📈 Tüketim Geçmişi · Test Dönemi · Gelecek 6 Ay</div>', unsafe_allow_html=True)

ts_train   = t_res.get("ts_train", [])
y_test     = t_res.get("y_test",  [])
y_pred     = t_res.get("y_pred_test", [])
tahminler  = t_res.get("tahminler", [])
tum_ml_pred= t_res.get("tum_ml_pred", {})
sampiyon_m = t_res.get("sampiyon","RF")

fig = go.Figure()

# Eğitim verisi
if ts_train:
    fig.add_trace(go.Scatter(
        x=ETIKET[:len(ts_train)], y=ts_train,
        name="Gerçek (Eğitim)", mode="lines",
        line=dict(color="#1E4D8C", width=2),
        hovertemplate="%{x}: <b>%{y:,.0f}</b><extra></extra>",
    ))

# Test gerçek
if y_test:
    fig.add_trace(go.Scatter(
        x=ETIKET[30:30+len(y_test)], y=y_test,
        name="Gerçek (Test)", mode="lines+markers",
        line=dict(color="#1E4D8C", width=2.5, dash="dot"),
        marker=dict(size=7), hovertemplate="%{x}: <b>%{y:,.0f}</b><extra></extra>",
    ))

# Tüm ML modellerin test tahminleri (ince çizgi)
for m_adi, m_pred in tum_ml_pred.items():
    is_samp = m_adi == sampiyon_m
    fig.add_trace(go.Scatter(
        x=ETIKET[30:30+len(m_pred)], y=m_pred,
        name=f"{m_adi} (Test)",
        mode="lines", opacity=1.0 if is_samp else 0.4,
        line=dict(color=RENKLER.get(m_adi,"#888"),
                  width=3 if is_samp else 1.5,
                  dash="solid" if is_samp else "dot"),
        hovertemplate="%{x}: <b>%{y:,.0f}</b><extra></extra>",
    ))

# Gelecek tahmin (şampiyon)
if tahminler:
    gel_x = ETIKET[36:36+len(tahminler)]
    fig.add_trace(go.Scatter(
        x=gel_x, y=tahminler,
        name=f"Tahmin ({sampiyon_m})",
        mode="lines+markers",
        line=dict(color=RENKLER.get(sampiyon_m,"#C8102E"), width=3, dash="dash"),
        marker=dict(size=9, symbol="diamond"),
        hovertemplate="%{x}: <b>%{y:,.0f}</b><extra></extra>",
    ))

# Ayrım çizgileri
for idx, label in [(30, "← Eğitim | Test →"), (36, "← Test | Tahmin →")]:
    if idx < len(ETIKET):
        fig.add_vline(x=ETIKET[idx], line_dash="dot", line_color="#F39200",
                      line_width=1.5)
        fig.add_annotation(x=ETIKET[idx], text=label, showarrow=False,
                           font=dict(size=9, color="#F39200"), yref="paper",
                           yanchor="bottom", y=0.02)

fig.update_layout(
    height=400, margin=dict(l=20,r=20,t=20,b=20),
    plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                font=dict(size=10)),
    xaxis=dict(showgrid=False, tickangle=-40, tickfont=dict(size=9)),
    yaxis=dict(showgrid=True, gridcolor="#F0F0F0", title="Tüketim (adet)", tickformat=","),
)
st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# AKSİYON UYARISI
# ──────────────────────────────────────────────────────────────────
renk  = uyari.get("renk","yesil")
mesaj = uyari.get("mesaj","")
st.markdown(f'<div class="aksiyon-{renk}">{mesaj}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# OPTİMAL STOK POLİTİKASI KARTLARI
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="bolum">📦 Optimal Stok Politikası (Grid Search + Optuna)</div>', unsafe_allow_html=True)

opt_Q  = o_res["optimal_Q"]
opt_r  = o_res["optimal_r"]
opt_SS = o_res["optimal_SS"]
hizmet = o_res["hizmet_duzeyi"]*100
Q_eoq  = o_res["Q_eoq"]
r_eoq  = o_res["r_eoq"]
SS_eoq = o_res["SS_eoq"]

# Model metrikleri kartları
st.markdown("<p style='font-size:11px;font-weight:700;color:#7B2D8B;margin:4px 0;'>MODEL PERFORMANS (PARÇA BAZLI TEST)</p>", unsafe_allow_html=True)
km1,km2,km3,km4 = st.columns(4)
samp_m = t_res["ml_metrikler"].get(sampiyon_m, {})
with km1: st.markdown(KART.format(c="MAE",  b="MAE",  d=f"{samp_m.get('MAE',0):,.0f}",  a="adet/ay (test)"), unsafe_allow_html=True)
with km2: st.markdown(KART.format(c="RMSE", b="RMSE", d=f"{samp_m.get('RMSE',0):,.0f}", a="adet/ay (test)"), unsafe_allow_html=True)
with km3: st.markdown(KART.format(c="MAPE", b="MAPE", d=f"{samp_m.get('MAPE',0):.1f}%", a="ortalama hata"), unsafe_allow_html=True)
with km4: st.markdown(KART.format(c="HZ",   b="Şampiyon", d=sampiyon_m, a=f"Segment: {seg_}"), unsafe_allow_html=True)

# Önerilen sistem
st.markdown("<p style='font-size:11px;font-weight:700;color:#C8102E;margin:12px 0 4px;'>ÖNERİLEN SİSTEM (ML + Grid Search + Optuna)</p>", unsafe_allow_html=True)
kc1,kc2,kc3,kc4 = st.columns(4)
with kc1: st.markdown(KART.format(c="Q",  b="Sipariş Miktarı (Q*)", d=f"{opt_Q:,}",     a="adet / sipariş"), unsafe_allow_html=True)
with kc2: st.markdown(KART.format(c="r",  b="Yeniden Sipariş (r*)", d=f"{opt_r:,}",     a="adet (ROP)"),     unsafe_allow_html=True)
with kc3: st.markdown(KART.format(c="SS", b="Emniyet Stoğu (SS*)",  d=f"{opt_SS:,}",    a="adet (z=1.65)"),  unsafe_allow_html=True)
with kc4: st.markdown(KART.format(c="HZ", b="Hizmet Düzeyi",        d=f"{hizmet:.1f}%", a="analitik"),       unsafe_allow_html=True)

# EOQ referans
st.markdown("<p style='font-size:11px;font-weight:700;color:#F39200;margin:12px 0 4px;'>EOQ KLASİK REFERANS (√2SD/h)</p>", unsafe_allow_html=True)
ke1,ke2,ke3,ke4 = st.columns(4)
with ke1: st.markdown(KART.format(c="Q",  b="EOQ Sipariş Miktarı", d=f"{Q_eoq:,}",  a="√(2SD/h)"),       unsafe_allow_html=True)
with ke2: st.markdown(KART.format(c="r",  b="EOQ Yeniden Sipariş", d=f"{r_eoq:,}",  a="μ_L + 1.65·σ_L"), unsafe_allow_html=True)
with ke3: st.markdown(KART.format(c="SS", b="EOQ Emniyet Stoğu",   d=f"{SS_eoq:,}", a="1.65 × σ_L"),     unsafe_allow_html=True)
with ke4: st.markdown(KART.format(c="HZ", b="EOQ Hizmet Hedefi",   d="%95.0",       a="z = 1.65"),       unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# TASARRUF BANNER
# ──────────────────────────────────────────────────────────────────
tl   = o_res["tasarruf_tl"]
oran = o_res["tasarruf_oran"]
yeni = o_res["yeni_maliyet"]
mev  = o_res["mevcut_maliyet"]
rng  = "linear-gradient(135deg,#00843D,#00A84F)" if tl>=0 else "linear-gradient(135deg,#C8102E,#E01535)"
ikon = "✅" if tl>=0 else "⚠️"
lbl  = "Aylık Tasarruf" if tl>=0 else "Maliyet Artışı"

st.markdown(f"""
<div style="background:{rng};border-radius:12px;padding:18px 26px;color:white;
     display:flex;justify-content:space-between;align-items:center;
     box-shadow:0 4px 16px rgba(0,0,0,.15);margin:14px 0;">
  <div>
    <div style="font-size:12px;opacity:.85;">{ikon} {lbl}</div>
    <div style="font-size:30px;font-weight:900;">{abs(tl):,.2f} TL/ay</div>
    <div style="font-size:11px;opacity:.75;">Yıllık: <b>{abs(tl)*12:,.0f} TL</b></div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:12px;opacity:.85;">Tasarruf Oranı</div>
    <div style="font-size:40px;font-weight:900;">{abs(oran):.1f}%</div>
    <div style="font-size:11px;opacity:.75;">Mevcut: {mev:,.0f} → Yeni: {yeni:,.0f} TL/ay</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# MALİYET KARŞILAŞTIRMA TABLOSU
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="bolum">💰 Maliyet Karşılaştırması: Mevcut vs. EOQ vs. Önerilen</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📋 Detaylı Tablo", "📊 Bar Grafik"])

kar_df = pd.DataFrame({
    "Maliyet Kalemi":           ["Elde Tutma","Sipariş","Stoksuz Kalma","TOPLAM"],
    "Mevcut Sistem":            [o_res["elde_tutma_mev"],o_res["siparis_maliyet_mev"],
                                  o_res["stoksuz_maliyet_mev"],o_res["mevcut_maliyet"]],
    "EOQ Klasik":               [o_res["et_eoq"],o_res["si_eoq"],o_res["sk_eoq"],o_res["tc_eoq"]],
    "Önerilen (ML+Grid+Optuna)":[o_res["elde_tutma_yeni"],o_res["siparis_maliyet_yeni"],
                                  o_res["stoksuz_maliyet_yeni"],o_res["yeni_maliyet"]],
})
kar_df["Tasarruf (TL/ay)"] = (kar_df["Mevcut Sistem"]-kar_df["Önerilen (ML+Grid+Optuna)"]).round(2)
kar_df["Tasarruf (%)"]     = (kar_df["Tasarruf (TL/ay)"]/kar_df["Mevcut Sistem"].replace(0,np.nan)*100).round(1)

with tab1:
    display = kar_df.copy()
    for c in ["Mevcut Sistem","EOQ Klasik","Önerilen (ML+Grid+Optuna)","Tasarruf (TL/ay)"]:
        display[c] = display[c].apply(lambda x: f"{x:,.2f} TL")
    display["Tasarruf (%)"] = display["Tasarruf (%)"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    st.dataframe(display, hide_index=True, use_container_width=True)

with tab2:
    kategoriler = ["Elde Tutma","Sipariş","Stoksuz Kalma"]
    mv_vals = kar_df.iloc[:3]["Mevcut Sistem"].tolist()
    eo_vals = kar_df.iloc[:3]["EOQ Klasik"].tolist()
    yn_vals = kar_df.iloc[:3]["Önerilen (ML+Grid+Optuna)"].tolist()
    mev_renk = ["#1E4D8C","#2D6FAE","#5B93C5"]
    eoq_renk = ["#F39200","#F5A623","#F7C46A"]
    yni_renk = ["#C8102E","#E01535","#F05070"]

    fig2 = go.Figure()
    for i,kat in enumerate(kategoriler):
        fig2.add_trace(go.Bar(name=f"{kat} (Mevcut)", x=["Mevcut"], y=[mv_vals[i]],
                               marker_color=mev_renk[i], text=f"{mv_vals[i]:,.0f}", textposition="inside"))
        fig2.add_trace(go.Bar(name=f"{kat} (EOQ)",    x=["EOQ"],    y=[eo_vals[i]],
                               marker_color=eoq_renk[i], text=f"{eo_vals[i]:,.0f}", textposition="inside"))
        fig2.add_trace(go.Bar(name=f"{kat} (Önerilen)",x=["ML+Optuna"],y=[yn_vals[i]],
                               marker_color=yni_renk[i], text=f"{yn_vals[i]:,.0f}", textposition="inside"))
    fig2.update_layout(barmode="stack", height=380,
                       plot_bgcolor="white", paper_bgcolor="white",
                       yaxis=dict(title="TL/ay",gridcolor="#F0F0F0"),
                       xaxis=dict(showgrid=False),
                       legend=dict(orientation="h",yanchor="bottom",y=1.01),
                       margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# PORTFÖY GENEL BAKIŞ
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="bolum">📊 Portföy Genel Bakış</div>', unsafe_allow_html=True)

pc1,pc2,pc3 = st.columns(3)

with pc1:
    if not batch.empty and "Sampiyon" in batch.columns:
        sc = batch["Sampiyon"].value_counts().reset_index()
        sc.columns = ["Model","Sayı"]
        fig_s = px.pie(sc, values="Sayı", names="Model", title="Şampiyon Model Dağılımı",
                       color="Model", color_discrete_map=RENKLER, hole=0.4)
        fig_s.update_layout(height=260,margin=dict(l=5,r=5,t=40,b=5),
                             plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fig_s, use_container_width=True)

with pc2:
    abc_seg = abc_df["Segment"].value_counts().reset_index()
    abc_seg.columns = ["Segment","Sayı"]
    fig_seg = px.bar(abc_seg, x="Segment", y="Sayı", title="ABC/XYZ Segment Dağılımı",
                     color="Segment", color_discrete_sequence=px.colors.qualitative.Set2)
    fig_seg.update_layout(height=260,margin=dict(l=5,r=5,t=40,b=5),
                           plot_bgcolor="white",paper_bgcolor="white",
                           showlegend=False,xaxis=dict(showgrid=False),
                           yaxis=dict(gridcolor="#F0F0F0"))
    st.plotly_chart(fig_seg, use_container_width=True)

with pc3:
    if not batch.empty and "MAPE" in batch.columns:
        mape_dist = batch["MAPE"].dropna()
        fig_mape = px.histogram(mape_dist, nbins=30, title="MAPE Dağılımı (Tüm Parçalar)",
                                color_discrete_sequence=["#1E4D8C"])
        fig_mape.update_layout(height=260,margin=dict(l=5,r=5,t=40,b=5),
                                plot_bgcolor="white",paper_bgcolor="white",
                                xaxis_title="MAPE (%)",yaxis=dict(gridcolor="#F0F0F0"),
                                showlegend=False)
        st.plotly_chart(fig_mape, use_container_width=True)

st.divider()
st.markdown("<div style='text-align:center;color:#999;font-size:11px;'>ENM412 | MAN Türkiye A.Ş. | Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN | 2024-2025</div>", unsafe_allow_html=True)
