# app.py
import re
from io import BytesIO
import pandas as pd
import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="Reservatórios – Consulta Rápida", layout="wide", page_icon="💧")

# =========================
# Estilo (tons de azul)
# =========================
st.markdown("""
<style>
:root{
  --azul1:#0f3d68; --azul2:#135e96; --azul3:#e9f3ff; --azul4:#f6faff;
}
html, body, [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, var(--azul4) 0%, #ffffff 60%); }
h1,h2,h3 { color: var(--azul1); }
.kpi { background: linear-gradient(180deg, #ffffff, var(--azul3)); border: 1px solid rgba(19,94,150,.15);
      box-shadow: 0 6px 18px rgba(19,94,150,.08); border-radius: 16px; padding: 18px 16px; }
.kpi .label { font-size: .85rem; color: #355b7a; margin-bottom: 4px; }
.kpi .value { font-size: 1.6rem; font-weight: 700; color: var(--azul2); }
.info { background: #ffffff; border-left: 4px solid var(--azul2); border-radius: 10px; padding: 12px 14px; color:#234; margin-bottom: 10px; }
[data-testid="stDataFrame"]  { border-radius: 12px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# =========================
# Utilitários de número (padrão BR)
# =========================
def parse_br_series(s: pd.Series) -> pd.Series:
    """Converte '1.234,56' -> 1234.56; remove %; lida com nulos."""
    return (s.astype(str)
             .str.replace(r"\s+", "", regex=True)
             .str.replace("%", "", regex=False)
             .str.replace(".", "", regex=False)
             .str.replace(",", ".", regex=False)
             .pipe(pd.to_numeric, errors="coerce"))

def fmt_br_2casas(x):
    if pd.isna(x): return "—"
    return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_br_inteiro(x):
    if pd.isna(x): return "—"
    return f"{float(x):,.0f}".replace(",", ".")

def fmt_br_pct(x):
    if pd.isna(x): return "—"
    return f"{float(x):,.2f} %".replace(",", "X").replace(".", ",").replace("X", ".")

def parse_br_number(txt: str) -> float | None:
    """'143,22' -> 143.22 | '1.234,5' -> 1234.5 | '' -> None"""
    if txt is None: return None
    s = str(txt).strip()
    if s == "": return None
    s = s.replace(" ", "").replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

# =========================
# Download dos dados
# =========================
def _extract_file_id_and_gid(url: str):
    m = re.search(r"/d/([a-zA-Z0-9-_]+)/", url)
    if not m: return None, None
    file_id = m.group(1)
    gid_match = re.search(r"[?&#]gid=(\d+)", url)
    gid = gid_match.group(1) if gid_match else "0"
    return file_id, gid

def build_possible_csv_urls(sheet_url: str):
    file_id, gid = _extract_file_id_and_gid(sheet_url)
    if not file_id: return [sheet_url]
    return [
        f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&gid={gid}",
        f"https://docs.google.com/spreadsheets/d/{file_id}/gviz/tq?tqx=out:csv&gid={gid}",
    ]

@st.cache_data(ttl=600, show_spinner="Carregando dados da planilha…")
def load_data(sheet_url: str):
    headers = {"User-Agent": "Mozilla/5.0 (Streamlit)"}
    last_err = None
    for u in build_possible_csv_urls(sheet_url):
        try:
            r = requests.get(u, headers=headers, allow_redirects=True, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(BytesIO(r.content), dtype=str)
            df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
    # fallback local
    try:
        df = pd.read_csv("/mnt/data/tabela_banabuiu_base.csv", dtype=str)
        df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
        return df
    except Exception:
        pass
    raise RuntimeError(f"Não foi possível baixar o CSV do Google Sheets. Último erro: {last_err}")

# =========================
# Carrega dados
# =========================
SHEET_URL = "https://docs.google.com/spreadsheets/d/1pavHCyfMX3fJvliRi_IseO-oUajzSKs302z0MQNcVnk/edit?usp=sharing"
df_raw = load_data(SHEET_URL)

st.title("💧 Consulta de Níveis e Volumes")
st.markdown(
    """
<div class="info">
  <b>Como usar:</b> escolha um <i>Reservatório</i> no filtro e, se quiser,
  preencha <b>Barrote</b>, <b>Régua (cm)</b> e digite <b>Cota (cm)</b> (ex.: 143,22).
  Os cartões exibem <i>Volume (m³)</i> e <i>Percentual</i> da primeira linha dos dados filtrados.
</div>
""", unsafe_allow_html=True
)

# =========================
# Detecta colunas originais
# =========================
reservatorio_col = next((c for c in df_raw.columns if c.strip().lower().startswith("reservat")), None)
col_barrote      = next((c for c in df_raw.columns if c.lower().startswith("barrote")), None)
col_regua        = next((c for c in df_raw.columns if "régua" in c.lower() or "regua" in c.lower()), None)
col_cota_cm_col  = next((c for c in df_raw.columns if re.sub(r"\s+", "", c.lower()) in ["cota(cm)","cotacm","cota cm"]), None)
col_cota_m_col   = next((c for c in df_raw.columns if re.sub(r"\s+", "", c.lower()) in ["cota(m)","cotam","cota m"]), None)
col_vol          = next((c for c in df_raw.columns if "volume (m3)" in c.lower() or "volume (m³)" in c.lower()), None)
col_pct          = next((c for c in df_raw.columns if "percentual" in c.lower()), None)

if not reservatorio_col:
    st.error("Não encontrei a coluna 'Reservatório' na planilha."); st.stop()
if not all([col_vol, col_pct]):
    st.error("Não encontrei as colunas 'Volume (m3)' e/ou 'Percentual'."); st.stop()

# =========================
# Cria colunas normalizadas para filtro (sem mexer na exibição)
# =========================
df_num = df_raw.copy()

if col_barrote:
    df_num["__barrote_num"] = parse_br_series(df_num[col_barrote])
if col_regua:
    df_num["__regua_num"] = parse_br_series(df_num[col_regua])

# Cota: sempre em centímetros para filtro
if col_cota_cm_col:
    cota_cm_num = parse_br_series(df_num[col_cota_cm_col])
elif col_cota_m_col:
    cota_cm_num = parse_br_series(df_num[col_cota_m_col]) * 100.0
else:
    cota_cm_num = pd.Series(np.nan, index=df_num.index, dtype=float)

df_num["__cota_cm_num"] = cota_cm_num
# Versão string BR (2 casas) para casar com texto da planilha
df_num["__cota_cm_str_br"] = df_num["__cota_cm_num"].round(2).apply(fmt_br_2casas)

# Outras colunas numéricas para KPI
df_num["__vol_num"] = parse_br_series(df_num[col_vol]) if col_vol else np.nan
df_num["__pct_num"] = parse_br_series(df_num[col_pct]) if col_pct else np.nan

# =========================
# Filtros (widgets)
# =========================
reservatorios = ["Todos"] + sorted([r for r in df_raw[reservatorio_col].dropna().astype(str).unique()])
sel_res = st.selectbox("Reservatório", options=reservatorios, index=0)

c1, c2, c3 = st.columns(3)
with c1:
    val_barrote = st.number_input("Barrote", value=None, step=1.0, format="%.0f")
with c2:
    val_regua   = st.number_input("Régua (cm)", value=None, step=1.0, format="%.0f")
with c3:
    val_cota_cm_txt = st.text_input("Cota (cm)", value="", placeholder="ex.: 143,22")

# =========================
# Aplica filtros (numérico + string BR)
# =========================
filtered_idx = pd.Series(True, index=df_num.index)

if sel_res != "Todos":
    filtered_idx &= (df_raw[reservatorio_col].astype(str) == sel_res)

tol_abs = 0.005  # tolerância ~ meio centésimo de cm

if col_barrote and val_barrote is not None:
    filtered_idx &= np.isfinite(df_num["__barrote_num"]) & np.isclose(df_num["__barrote_num"], float(val_barrote), atol=1e-9, rtol=0)

if col_regua and val_regua is not None:
    filtered_idx &= np.isfinite(df_num["__regua_num"]) & np.isclose(df_num["__regua_num"], float(val_regua), atol=1e-9, rtol=0)

typed_cota_cm_num = parse_br_number(val_cota_cm_txt)
if typed_cota_cm_num is not None:
    # 1) Comparação numérica (com tolerância)
    mask_num = np.isfinite(df_num["__cota_cm_num"]) & np.isclose(df_num["__cota_cm_num"], typed_cota_cm_num, atol=tol_abs, rtol=0)
    # 2) Comparação por string BR (2 casas) — cobre quando a planilha guarda como texto
    typed_cota_cm_str = fmt_br_2casas(typed_cota_cm_num)  # ex.: "143,22"
    mask_str = (df_num["__cota_cm_str_br"] == typed_cota_cm_str)

    # 3) Se a base só tiver Cota (m), também aceito que a pessoa digite em m por engano (m->cm)
    mask_extra = False
    if col_cota_m_col and not col_cota_cm_col:
        typed_as_m_num = typed_cota_cm_num / 100.0  # cm->m
        mask_extra = np.isfinite(df_num["__cota_cm_num"]) & np.isclose(df_num["__cota_cm_num"], typed_as_m_num * 100.0, atol=tol_abs, rtol=0)

    filtered_idx &= (mask_num | mask_str | mask_extra)

filtered_raw = df_raw[filtered_idx].copy()
filtered_num = df_num[filtered_idx].copy()

# =========================
# KPIs (1ª linha do resultado filtrado)
# =========================
if len(filtered_num):
    first = filtered_num.iloc[0]
    vol_val = first["__vol_num"]
    pct_val = first["__pct_num"]
else:
    vol_val = np.nan
    pct_val = np.nan

k1, k2, k3 = st.columns([1,1,2])
with k1:
    st.markdown('<div class="kpi"><div class="label">Volume (m³)</div>'
                f'<div class="value">{fmt_br_inteiro(vol_val)}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi"><div class="label">Percentual</div>'
                f'<div class="value">{fmt_br_pct(pct_val)}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(
        f'<div class="kpi"><div class="label">Registros encontrados</div>'
        f'<div class="value">{len(filtered_raw):,}</div></div>'.replace(",", "."),
        unsafe_allow_html=True
    )

# =========================
# Tabela (copia da planilha, sem converter Cota)
# =========================
desired_cols = ["Reservatório", "Barrote", "Régua (cm)", "Cota (m)", "Volume (m3)", "Percentual"]
available = set(filtered_raw.columns)
if "Cota (m)" not in available and col_cota_cm_col:
    desired_cols = ["Reservatório", "Barrote", "Régua (cm)", col_cota_cm_col, "Volume (m3)", "Percentual"]

cols_to_show = [c for c in desired_cols if c in available]
df_view_raw = filtered_raw[cols_to_show].copy()

# Formatação BR apenas para exibir
def format_view(dfv: pd.DataFrame) -> pd.DataFrame:
    out = dfv.copy()
    if "Volume (m3)" in out.columns:
        out["Volume (m3)"] = parse_br_series(out["Volume (m3)"]).apply(fmt_br_inteiro)
    if "Percentual" in out.columns:
        out["Percentual"] = parse_br_series(out["Percentual"]).apply(fmt_br_pct)
    if "Cota (m)" in out.columns:
        out["Cota (m)"] = parse_br_series(out["Cota (m)"]).apply(fmt_br_2casas)
    if col_cota_cm_col and col_cota_cm_col in out.columns:
        out[col_cota_cm_col] = parse_br_series(out[col_cota_cm_col]).apply(fmt_br_2casas)
    if "Barrote" in out.columns:
        out["Barrote"] = parse_br_series(out["Barrote"]).map(lambda x: f"{x:,.0f}".replace(",", ".") if pd.notna(x) else "—")
    if "Régua (cm)" in out.columns:
        out["Régua (cm)"] = parse_br_series(out["Régua (cm)"]).map(lambda x: f"{x:,.0f}".replace(",", ".") if pd.notna(x) else "—")
    return out

df_view = format_view(df_view_raw)

st.subheader("Tabela filtrada")
column_config = {
    "Barrote":     st.column_config.TextColumn("Barrote"),
    "Régua (cm)":  st.column_config.TextColumn("Régua (cm)"),
    "Cota (m)":    st.column_config.TextColumn("Cota (m)"),
    "Volume (m3)": st.column_config.TextColumn("Volume (m3)"),
    "Percentual":  st.column_config.TextColumn("Percentual"),
}
if col_cota_cm_col and col_cota_cm_col in df_view.columns:
    column_config[col_cota_cm_col] = st.column_config.TextColumn(col_cota_cm_col)

st.dataframe(
    df_view,
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
)
