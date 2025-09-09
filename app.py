# app.py
import re
import math
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
html, body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, var(--azul4) 0%, #ffffff 60%);
}
h1,h2,h3 { color: var(--azul1); }
.kpi {
  background: linear-gradient(180deg, #ffffff, var(--azul3));
  border: 1px solid rgba(19,94,150,.15);
  box-shadow: 0 6px 18px rgba(19,94,150,.08);
  border-radius: 16px; padding: 18px 16px;
}
.kpi .label { font-size: .85rem; color: #355b7a; margin-bottom: 4px; }
.kpi .value { font-size: 1.6rem; font-weight: 700; color: var(--azul2); }
.info {
  background: #ffffff; border-left: 4px solid var(--azul2);
  border-radius: 10px; padding: 12px 14px; color:#234; margin-bottom: 10px;
}
[data-testid="stDataFrame"]  { border-radius: 12px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers para URL do Sheets
# =========================
def _extract_file_id_and_gid(url: str):
    m = re.search(r"/d/([a-zA-Z0-9-_]+)/", url)
    if not m:
        return None, None
    file_id = m.group(1)
    gid_match = re.search(r"[?&#]gid=(\d+)", url)
    gid = gid_match.group(1) if gid_match else "0"
    return file_id, gid

def build_possible_csv_urls(sheet_url: str):
    file_id, gid = _extract_file_id_and_gid(sheet_url)
    if not file_id:
        return [sheet_url]  # tenta como veio
    return [
        # 1) Export oficial (requer "Qualquer pessoa com o link: Leitor")
        f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&gid={gid}",
        # 2) Endpoint gviz (funciona bem para planilhas públicas)
        f"https://docs.google.com/spreadsheets/d/{file_id}/gviz/tq?tqx=out:csv&gid={gid}",
        # 3) Em caso de URL já publicada em CSV (mantém se o usuário colar pronto)
        sheet_url if sheet_url.endswith("output=csv") or sheet_url.endswith("format=csv") else None,
    ]

@st.cache_data(ttl=600, show_spinner="Carregando dados da planilha…")
def load_data(sheet_url: str):
    last_err = None
    source_used = None
    headers = {"User-Agent": "Mozilla/5.0 (Streamlit)"}

    for u in [u for u in build_possible_csv_urls(sheet_url) if u]:
        try:
            r = requests.get(u, headers=headers, allow_redirects=True, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(BytesIO(r.content), dtype=str)
            # normaliza nomes
            df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
            # tentativa de numerificação básica
            for c in df.columns:
                if re.search(r"(barrote|r[ée]gua|cota|volume|percentual|\(cm\)|\(m3\)|\(m³\))", c, re.IGNORECASE):
                    s = (df[c].astype(str)
                            .str.replace(".", "", regex=False)
                            .str.replace(",", ".", regex=False)
                            .str.replace("%", "", regex=False)
                            .str.strip())
                    df[c] = pd.to_numeric(s, errors="ignore")
            source_used = u
            return df, source_used
        except Exception as e:
            last_err = e

    # Fallback local, se existir (útil no Streamlit Community)
    try:
        local_path = "/mnt/data/tabela_banabuiu_base.csv"
        df = pd.read_csv(local_path, dtype=str)
        df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
        for c in df.columns:
            if re.search(r"(barrote|r[ée]gua|cota|volume|percentual|\(cm\)|\(m3\)|\(m³\))", c, re.IGNORECASE):
                s = (df[c].astype(str)
                        .str.replace(".", "", regex=False)
                        .str.replace(",", ".", regex=False)
                        .str.replace("%", "", regex=False)
                        .str.strip())
                df[c] = pd.to_numeric(s, errors="ignore")
        source_used = local_path
        return df, source_used
    except Exception:
        pass

    raise RuntimeError(
        f"Não foi possível baixar o CSV do Google Sheets. "
        f"Último erro recebido: {last_err}"
    )

def fmt_number(x, is_percent=False):
    if pd.isna(x):
        return "—"
    if is_percent:
        return f"{float(x):,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{float(x):,.0f}".replace(",", ".")

# =========================
# Carrega dados
# =========================
SHEET_URL = "https://docs.google.com/spreadsheets/d/1pavHCyfMX3fJvliRi_IseO-oUajzSKs302z0MQNcVnk/edit?usp=sharing"
df_raw, data_source = load_data(SHEET_URL)

st.title("💧 Consulta de Níveis e Volumes")
st.markdown(
    """
<div class="info">
  <b>Como usar:</b> escolha um <i>Reservatório</i> no filtro e, se quiser,
  preencha <b>Barrote</b>, <b>Régua (cm)</b> e <b>Cota (cm)</b> para localizar a(s) linha(s) exata(s).
  Os cartões exibem <i>Volume (m3)</i> e <i>Percentual</i> do resultado filtrado.
</div>
""",
    unsafe_allow_html=True,
)
st.caption(f"Fonte de dados: {data_source}")

# =========================
# Filtro por Reservatório
# =========================
reservatorio_col = next((c for c in df_raw.columns if c.strip().lower().startswith("reservat")), None)
if not reservatorio_col:
    st.error("Não encontrei a coluna 'Reservatório' na planilha.")
    st.stop()

reservatorios = ["Todos"] + sorted([r for r in df_raw[reservatorio_col].dropna().astype(str).unique()])
sel_res = st.selectbox("Reservatório", options=reservatorios, index=0)

df = df_raw.copy()
if sel_res != "Todos":
    df = df[df[reservatorio_col].astype(str) == sel_res]

# =========================
# Entradas do usuário
# =========================
c1, c2, c3 = st.columns(3)
with c1:
    val_barrote = st.number_input("Barrote", value=None, step=1.0, format="%.0f")
with c2:
    val_regua = st.number_input("Régua (cm)", value=None, step=1.0, format="%.0f")
with c3:
    val_cota = st.number_input("Cota (cm)", value=None, step=1.0, format="%.0f")

# mapear nomes possíveis
col_barrote = next((c for c in df.columns if c.lower().startswith("barrote")), None)
col_regua  = next((c for c in df.columns if "régua" in c.lower() or "regua" in c.lower()), None)
col_cota   = next((c for c in df.columns if c.lower().startswith("cota")), None)
col_vol    = next((c for c in df.columns if "volume (m3)" in c.lower() or "volume (m³)" in c.lower()), None)
col_pct    = next((c for c in df.columns if "percentual" in c.lower()), None)

if not all([col_vol, col_pct]):
    st.warning("Não encontrei as colunas 'Volume (m3)' e/ou 'Percentual'. Verifique os nomes na planilha.")
    st.stop()

# =========================
# Filtragem por valores (opcional)
# =========================
filtered = df.copy()
tol = 1e-6
if col_barrote and val_barrote is not None:
    filtered = filtered[np.isclose(pd.to_numeric(filtered[col_barrote], errors="coerce"), val_barrote, atol=tol, rtol=0)]
if col_regua and val_regua is not None:
    filtered = filtered[np.isclose(pd.to_numeric(filtered[col_regua], errors="coerce"), val_regua, atol=tol, rtol=0)]
if col_cota and val_cota is not None:
    filtered = filtered[np.isclose(pd.to_numeric(filtered[col_cota], errors="coerce"), val_cota, atol=tol, rtol=0)]

# =========================
# KPIs
# =========================
vol_val = filtered[col_vol].astype(float).mean() if len(filtered) else np.nan
pct_val = filtered[col_pct].astype(float).mean() if len(filtered) else np.nan

k1, k2, k3 = st.columns([1,1,2])
with k1:
    st.markdown('<div class="kpi"><div class="label">Volume (m³)</div>'
                f'<div class="value">{fmt_number(vol_val)}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi"><div class="label">Percentual</div>'
                f'<div class="value">{fmt_number(pct_val, is_percent=True)}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(
        f'<div class="kpi"><div class="label">Registros encontrados</div>'
        f'<div class="value">{len(filtered):,}</div></div>'.replace(",", "."),
        unsafe_allow_html=True
    )

# =========================
# Tabela
# =========================
st.subheader("Tabela filtrada")
st.dataframe(filtered, use_container_width=True)
