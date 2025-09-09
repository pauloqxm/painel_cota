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
# Helpers Sheets
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
            # numerificar colunas típicas
            for c in df.columns:
                if re.search(r"(barrote|r[ée]gua|cota|volume|percentual|\(cm\)|\(m3\)|\(m³\))", c, re.IGNORECASE):
                    s = (df[c].astype(str)
                           .str.replace(".", "", regex=False)
                           .str.replace(",", ".", regex=False)
                           .str.replace("%", "", regex=False)
                           .str.strip())
                    df[c] = pd.to_numeric(s, errors="ignore")
            return df
        except Exception as e:
            last_err = e

    # fallback local (opcional)
    try:
        df = pd.read_csv("/mnt/data/tabela_banabuiu_base.csv", dtype=str)
        df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
        for c in df.columns:
            if re.search(r"(barrote|r[ée]gua|cota|volume|percentual|\(cm\)|\(m3\)|\(m³\))", c, re.IGNORECASE):
                s = (df[c].astype(str)
                       .str.replace(".", "", regex=False)
                       .str.replace(",", ".", regex=False)
                       .str.replace("%", "", regex=False)
                       .str.strip())
                df[c] = pd.to_numeric(s, errors="ignore")
        return df
    except Exception:
        pass

    raise RuntimeError(f"Não foi possível baixar o CSV do Google Sheets. Último erro: {last_err}")

# =========================
# Formatações BR
# =========================
def fmt_br_2casas(x):
    if pd.isna(x): 
        return "—"
    return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_br_inteiro(x):
    if pd.isna(x): 
        return "—"
    return f"{float(x):,.0f}".replace(",", ".")

def fmt_br_pct(x):
    if pd.isna(x):
        return "—"
    return f"{float(x):,.2f} %".replace(",", "X").replace(".", ",").replace("X", ".")

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
  preencha <b>Barrote</b>, <b>Régua (cm)</b> e <b>Cota (cm)</b> para localizar a(s) linha(s) exata(s).
  Os cartões exibem <i>Volume (m³)</i> e <i>Percentual</i> do resultado filtrado.
</div>
""", unsafe_allow_html=True
)

# =========================
# Filtro por Reservatório
# =========================
reservatorio_col = next((c for c in df_raw.columns if c.strip().lower().startswith("reservat")), None)
if not reservatorio_col:
    st.error("Não encontrei a coluna 'Reservatório' na planilha."); st.stop()

reservatorios = ["Todos"] + sorted([r for r in df_raw[reservatorio_col].dropna().astype(str).unique()])
sel_res = st.selectbox("Reservatório", options=reservatorios, index=0)

df = df_raw.copy()
if sel_res != "Todos":
    df = df[df[reservatorio_col].astype(str) == sel_res]

# =========================
# Entradas do usuário
# =========================
c1, c2, c3 = st.columns(3)
with c1: val_barrote = st.number_input("Barrote", value=None, step=1.0, format="%.0f")
with c2: val_regua   = st.number_input("Régua (cm)", value=None, step=1.0, format="%.0f")
with c3: val_cota    = st.number_input("Cota (cm)", value=None, step=1.0, format="%.0f")

# nomes possíveis
col_barrote = next((c for c in df.columns if c.lower().startswith("barrote")), None)
col_regua   = next((c for c in df.columns if "régua" in c.lower() or "regua" in c.lower()), None)
col_cota_cm = next((c for c in df.columns if c.lower().startswith("cota")), None)
col_vol     = next((c for c in df.columns if "volume (m3)" in c.lower() or "volume (m³)" in c.lower()), None)
col_pct     = next((c for c in df.columns if "percentual" in c.lower()), None)

if not all([col_vol, col_pct]):
    st.warning("Não encontrei as colunas 'Volume (m3)' e/ou 'Percentual'."); st.stop()

# =========================
# Filtragem pelos valores
# =========================
filtered = df.copy()
tol = 1e-6
if col_barrote and val_barrote is not None:
    filtered = filtered[np.isclose(pd.to_numeric(filtered[col_barrote], errors="coerce"), val_barrote, atol=tol, rtol=0)]
if col_regua and val_regua is not None:
    filtered = filtered[np.isclose(pd.to_numeric(filtered[col_regua], errors="coerce"), val_regua, atol=tol, rtol=0)]
if col_cota_cm and val_cota is not None:
    filtered = filtered[np.isclose(pd.to_numeric(filtered[col_cota_cm], errors="coerce"), val_cota, atol=tol, rtol=0)]

# =========================
# KPIs (com BR e %)
# =========================
vol_val = pd.to_numeric(filtered[col_vol], errors="coerce").mean() if len(filtered) else np.nan
pct_val = pd.to_numeric(filtered[col_pct], errors="coerce").mean() if len(filtered) else np.nan

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
        f'<div class="value">{len(filtered):,}</div></div>'.replace(",", "."),
        unsafe_allow_html=True
    )

# =========================
# Tabela (somente colunas pedidas + formatação BR)
# =========================
data_num = {}
data_num["Reservatório"] = filtered[reservatorio_col].astype(str) if reservatorio_col in filtered.columns else np.nan
if col_barrote: data_num["Barrote"]     = pd.to_numeric(filtered[col_barrote], errors="coerce")
if col_regua:   data_num["Régua (cm)"]  = pd.to_numeric(filtered[col_regua], errors="coerce")
if col_cota_cm: data_num["Cota (m)"]    = pd.to_numeric(filtered[col_cota_cm], errors="coerce") / 100.0
if col_vol:     data_num["Volume (m3)"] = pd.to_numeric(filtered[col_vol], errors="coerce")
if col_pct:     data_num["Percentual"]  = pd.to_numeric(filtered[col_pct], errors="coerce")

df_view_num = pd.DataFrame(
    data_num,
    columns=["Reservatório","Barrote","Régua (cm)","Cota (m)","Volume (m3)","Percentual"]
)

# Exibição em BR:
df_view = df_view_num.copy()
if "Cota (m)" in df_view:
    df_view["Cota (m)"] = df_view_num["Cota (m)"].apply(fmt_br_2casas)
if "Volume (m3)" in df_view:
    df_view["Volume (m3)"] = df_view_num["Volume (m3)"].apply(fmt_br_inteiro)
if "Percentual" in df_view:
    df_view["Percentual"] = df_view_num["Percentual"].apply(fmt_br_pct)

st.subheader("Tabela filtrada")
st.dataframe(
    df_view,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Barrote":     st.column_config.NumberColumn("Barrote", format="%.0f"),
        "Régua (cm)":  st.column_config.NumberColumn("Régua (cm)", format="%.0f"),
        # As três abaixo são strings já formatadas em BR (vírgula/ponto/%):
        "Cota (m)":    st.column_config.TextColumn("Cota (m)"),
        "Volume (m3)": st.column_config.TextColumn("Volume (m3)"),
        "Percentual":  st.column_config.TextColumn("Percentual"),
    },
)
