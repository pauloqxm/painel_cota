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
# Formatações BR / utilitários
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

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def parse_br_number(txt: str) -> float | None:
    """Converte '143,22' ou '1.234,5' -> 143.22 / 1234.5. Retorna None se vazio/ inválido."""
    if txt is None: 
        return None
    s = str(txt).strip()
    if s == "": 
        return None
    s = s.replace(" ", "").replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

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
  preencha <b>Barrote</b>, <b>Régua (cm)</b> e digite <b>Cota (cm)</b> (ex.: 143,22) para localizar a(s) linha(s) exata(s).
  Os cartões exibem <i>Volume (m³)</i> e <i>Percentual</i> da primeira linha dos dados filtrados.
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
# Detecta colunas
# =========================
col_barrote = next((c for c in df.columns if c.lower().startswith("barrote")), None)
col_regua   = next((c for c in df.columns if "régua" in c.lower() or "regua" in c.lower()), None)
# Preferimos Cota (cm); se não existir, caímos para Cota (m)
col_cota_cm_col = next((c for c in df.columns if re.sub(r"\s+", "", c.lower()) in ["cota(cm)","cotacm","cota cm"]), None)
col_cota_m_col  = next((c for c in df.columns if re.sub(r"\s+", "", c.lower()) in ["cota(m)","cotam","cota m"]), None)
col_vol     = next((c for c in df.columns if "volume (m3)" in c.lower() or "volume (m³)" in c.lower()), None)
col_pct     = next((c for c in df.columns if "percentual" in c.lower()), None)

if not all([col_vol, col_pct]):
    st.warning("Não encontrei as colunas 'Volume (m3)' e/ou 'Percentual'."); st.stop()

# =========================
# Widgets de filtro
# =========================
c1, c2, c3 = st.columns(3)

with c1:
    val_barrote = st.number_input("Barrote", value=None, step=1.0, format="%.0f")

with c2:
    val_regua   = st.number_input("Régua (cm)", value=None, step=1.0, format="%.0f")

with c3:
    # Entrada DIGITÁVEL que aceita vírgula (ex.: 143,22)
    val_cota_cm_txt = st.text_input(
        "Cota (cm)",
        value="",
        placeholder="ex.: 143,22",
        help="Digite no mesmo padrão da planilha (vírgula como decimal, se houver)."
    )

# =========================
# Aplica filtros
# =========================
filtered = df.copy()
tol = 1e-6

if col_barrote and val_barrote is not None:
    filtered = filtered[np.isclose(to_num(filtered[col_barrote]), val_barrote, atol=tol, rtol=0)]

if col_regua and val_regua is not None:
    filtered = filtered[np.isclose(to_num(filtered[col_regua]), val_regua, atol=tol, rtol=0)]

# --- Filtro Cota (cm) com vírgula e robusto a unidade (cm vs m) ---
val_cota_cm = parse_br_number(val_cota_cm_txt)
if val_cota_cm is not None:
    if col_cota_cm_col:
        s = to_num(filtered[col_cota_cm_col])  # já em cm
        s_r2 = s.round(2)
        # usuário pode ter digitado em cm (143,22) ou m por engano (-> *100)
        targets = { round(val_cota_cm, 2), round(val_cota_cm * 100.0, 2) }
        filtered = filtered[s_r2.isin(targets)]
    elif col_cota_m_col:
        s_m = to_num(filtered[col_cota_m_col])          # em m
        s_cm_r2 = (s_m * 100.0).round(2)
        s_m_r2  = s_m.round(2)
        target_cm = round(val_cota_cm, 2)               # assume entrada em cm
        target_m  = round(val_cota_cm / 100.0, 2)       # fallback se entrada estiver em m
        mask = (s_cm_r2 == target_cm) | (s_m_r2 == target_m)
        filtered = filtered[mask]

# =========================
# KPIs (pegam a 1ª linha dos dados filtrados)
# =========================
if len(filtered):
    first_row = filtered.iloc[0]
    vol_val = to_num(first_row[col_vol])
    pct_val = to_num(first_row[col_pct])
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
        f'<div class="value">{len(filtered):,}</div></div>'.replace(",", "."),
        unsafe_allow_html=True
    )

# =========================
# Tabela (copia exatamente da planilha, sem converter Cota)
# =========================
desired_cols = ["Reservatório", "Barrote", "Régua (cm)", "Cota (m)", "Volume (m3)", "Percentual"]
available = {c: c for c in filtered.columns}

# Se não existir "Cota (m)" mas existir "Cota (cm)", usamos "Cota (cm)"
if "Cota (m)" not in available and col_cota_cm_col:
    desired_cols = ["Reservatório", "Barrote", "Régua (cm)", col_cota_cm_col, "Volume (m3)", "Percentual"]

cols_to_show = [c for c in desired_cols if c in available]
df_view_raw = filtered[cols_to_show].copy()

def format_view(dfv: pd.DataFrame) -> pd.DataFrame:
    out = dfv.copy()
    if "Volume (m3)" in out.columns:
        out["Volume (m3)"] = pd.to_numeric(out["Volume (m3)"], errors="coerce").apply(fmt_br_inteiro)
    if "Percentual" in out.columns:
        out["Percentual"] = pd.to_numeric(out["Percentual"], errors="coerce").apply(fmt_br_pct)
    if "Cota (m)" in out.columns:
        out["Cota (m)"] = pd.to_numeric(out["Cota (m)"], errors="coerce").apply(fmt_br_2casas)
    if col_cota_cm_col and col_cota_cm_col in out.columns:
        out[col_cota_cm_col] = pd.to_numeric(out[col_cota_cm_col], errors="coerce").apply(fmt_br_2casas)
    if "Barrote" in out.columns:
        out["Barrote"] = pd.to_numeric(out["Barrote"], errors="coerce").map(lambda x: f"{x:,.0f}".replace(",", ".") if pd.notna(x) else "—")
    if "Régua (cm)" in out.columns:
        out["Régua (cm)"] = pd.to_numeric(out["Régua (cm)"], errors="coerce").map(lambda x: f"{x:,.0f}".replace(",", ".") if pd.notna(x) else "—")
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
