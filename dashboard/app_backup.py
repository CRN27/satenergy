import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import geopandas as gpd
import requests
import joblib
import json
import zipfile
import re
import warnings

from io import StringIO
from datetime import datetime
from shapely.geometry import Point
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="SatEnergy — Sistema de Previsão Renovável",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FUNÇÕES DE CARREGAMENTO (com cache)
# ============================================================
@st.cache_data
def carregar_shapefile_brasil():
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    mundo = gpd.read_file(url)
    brasil = mundo[mundo["name"] == "Brazil"]
    return brasil

@st.cache_data
def carregar_ons():
    import glob
    arquivos = glob.glob("data/raw/FATOR*.csv")
    if not arquivos:
        return None
    df = pd.read_csv(arquivos[0], sep=";", encoding="utf-8")
    df["din_instante"] = pd.to_datetime(df["din_instante"])
    df["hora"] = df["din_instante"].dt.hour
    df["mes"]  = df["din_instante"].dt.month
    return df

@st.cache_data
def carregar_linhas_transmissao():
    kmz_path = "data/raw/[EPE] Linhas de Transmissão.kmz"
    try:
        with zipfile.ZipFile(kmz_path, "r") as kmz:
            for arquivo in kmz.namelist():
                if arquivo.endswith(".kml"):
                    kmz.extract(arquivo, "data/raw/era5/")
        gdf = gpd.read_file("data/raw/era5/doc.kml")
        gdf["Tens__o__kV_"] = pd.to_numeric(gdf["Tens__o__kV_"], errors="coerce")
        def extrair_sub(nome):
            match = re.search(r"LT \d+ kV (.+?) [-–] (.+?)(?:,|$)", str(nome))
            if match:
                return match.group(1).strip(), match.group(2).strip().split(",")[0].strip()
            return "Desconhecida", "Desconhecida"
        gdf[["sub_origem", "sub_destino"]] = gdf["Nome"].apply(lambda x: pd.Series(extrair_sub(x)))
        return gdf
    except:
        return None

@st.cache_data(ttl=600)
def carregar_focos_inpe():
    URL_INPE = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/10min/"
    try:
        r = requests.get(URL_INPE, verify=False, timeout=15)
        arquivos_csv = re.findall(r'href="(focos_[^"]+\.csv)"', r.text)
        dfs = []
        for arquivo in arquivos_csv[-18:]:
            r_csv = requests.get(f"{URL_INPE}{arquivo}", verify=False, timeout=15)
            df_temp = pd.read_csv(StringIO(r_csv.text))
            if len(df_temp) > 0:
                dfs.append(df_temp)
        if dfs:
            return pd.concat(dfs, ignore_index=True)
    except:
        pass
    return pd.DataFrame()

@st.cache_resource
def carregar_modelos():
    try:
        solar  = joblib.load("models/modelo_solar.pkl")
        eolico = joblib.load("models/modelo_eolico.pkl")
        return solar, eolico
    except:
        return None, None

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("https://img.icons8.com/color/96/satellite.png", width=80)
st.sidebar.title("🛰️ SatEnergy")
st.sidebar.caption("Sistema de Previsão de Geração Renovável via Dados Orbitais")
st.sidebar.divider()

pagina = st.sidebar.radio(
    "Navegação",
    ["🏠 Visão Geral", "🔥 Alertas de Risco", "⚡ Geração ONS", "🤖 Previsão ML"],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.caption(f"Atualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# ============================================================
# PÁGINA 1 — VISÃO GERAL
# ============================================================
if pagina == "🏠 Visão Geral":
    st.title("🛰️ SatEnergy")
    st.subheader("Sistema de Previsão de Geração Renovável via Dados Orbitais")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    df_ons = carregar_ons()
    modelo_solar, modelo_eolico = carregar_modelos()

    with col1:
        st.metric("Registros ONS", f"{len(df_ons):,}" if df_ons is not None else "N/A", "Fator de capacidade")
    with col2:
        n_solar = len(df_ons[df_ons["nom_tipousina"] == "Solar"]) if df_ons is not None else 0
        st.metric("Usinas Solares", f"{df_ons['nom_usina_conjunto'].nunique():,}" if df_ons is not None else "N/A", "no dataset")
    with col3:
        st.metric("Modelo Solar R²", "0.880", "+1.0% com NASA")
    with col4:
        st.metric("Modelo Eólico R²", "0.723", "XGBoost")

    st.divider()

    st.subheader("📊 Fontes de Dados")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        | Fonte | Dados | Status |
        |-------|-------|--------|
        | ONS | Fator de capacidade solar/eólica | ✅ Ativo |
        | INPE / GOES-19 | Focos de calor em tempo real | ✅ Ativo |
        | NASA POWER | Irradiância, temperatura, vento | ✅ Ativo |
        | Copernicus ERA5 | Vento a 10m e 100m | ✅ Ativo |
        | EPE | Linhas de transmissão | ✅ Ativo |
        | ANEEL SIGEL | Subestações de energia | ✅ Ativo |
        """)

    with col2:
        st.markdown("""
        ### 🎯 Modelos
        - **Modelo 1 — Previsão de Geração:** XGBoost treinado com dados ONS + NASA POWER + ERA5. R² Solar = 0.88, R² Eólico = 0.72
        - **Modelo 2 — Score de Risco:** Calcula distância de focos de calor (GOES-19) até linhas de transmissão e classifica risco em 4 níveis
        """)

# ============================================================
# PÁGINA 2 — ALERTAS DE RISCO
# ============================================================
elif pagina == "🔥 Alertas de Risco":
    st.title("🔥 Sistema de Alertas — Score de Risco Climático")
    st.caption("Focos de calor detectados pelo GOES-19 vs infraestrutura energética")
    st.divider()

    with st.spinner("🛰️ Buscando focos de calor do GOES-19..."):
        df_focos = carregar_focos_inpe()
        brasil   = carregar_shapefile_brasil()
        gdf_lt   = carregar_linhas_transmissao()

    if df_focos.empty:
        st.warning("Sem dados de focos disponíveis no momento.")
    else:
        brasil_shape = unary_union(brasil.geometry)
        geometry = [Point(xy) for xy in zip(df_focos["lon"], df_focos["lat"])]
        gdf_todos = gpd.GeoDataFrame(df_focos, geometry=geometry, crs="EPSG:4326")
        gdf_todos["dentro_brasil"] = gdf_todos.geometry.within(brasil_shape)
        df_focos_brasil = gdf_todos[gdf_todos["dentro_brasil"]].drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Focos detectados", len(df_focos_brasil), "últimas 3 horas")
        with col2:
            st.metric("Satélite", "GOES-19", "INPE")
        with col3:
            st.metric("Atualização", "10 min", "tempo quase-real")

        if gdf_lt is not None and len(df_focos_brasil) > 0:
            with st.spinner("Calculando distâncias..."):
                gdf_focos_proj = df_focos_brasil.to_crs("EPSG:31983")
                gdf_lt_proj    = gdf_lt.to_crs("EPSG:31983")

                def linha_mais_proxima(ponto):
                    dist = gdf_lt_proj.geometry.distance(ponto)
                    idx  = dist.idxmin()
                    return pd.Series({
                        "dist_km":      dist[idx] / 1000,
                        "linha_nome":   gdf_lt.loc[idx, "Nome"],
                        "tensao_kv":    gdf_lt.loc[idx, "Tens__o__kV_"],
                        "sub_origem":   gdf_lt.loc[idx, "sub_origem"],
                        "sub_destino":  gdf_lt.loc[idx, "sub_destino"],
                    })

                df_dist = gdf_focos_proj.geometry.apply(linha_mais_proxima)
                gdf_focos_proj = pd.concat([gdf_focos_proj.reset_index(drop=True), df_dist.reset_index(drop=True)], axis=1)

                def risco(d):
                    if d <= 10:   return "🔴 CRÍTICO"
                    elif d <= 50:  return "🟠 ALTO"
                    elif d <= 100: return "🟡 MÉDIO"
                    else:          return "🟢 BAIXO"

                gdf_focos_proj["risco"] = gdf_focos_proj["dist_km"].apply(risco)

            st.subheader("📊 Distribuição de Risco")
            contagem = gdf_focos_proj["risco"].value_counts()
            col1, col2, col3, col4 = st.columns(4)
            for col, (nivel, cor) in zip([col1, col2, col3, col4],
                [("🔴 CRÍTICO", "red"), ("🟠 ALTO", "orange"), ("🟡 MÉDIO", "yellow"), ("🟢 BAIXO", "green")]):
                with col:
                    st.metric(nivel, contagem.get(nivel, 0))

            st.subheader("🗺️ Mapa de Risco")
            fig, ax = plt.subplots(figsize=(12, 13))
            brasil.plot(ax=ax, color="#d4e6b5", edgecolor="black", linewidth=0.8)

            lt_750   = gdf_lt[gdf_lt["Tens__o__kV_"] >= 750]
            lt_500   = gdf_lt[(gdf_lt["Tens__o__kV_"] >= 500) & (gdf_lt["Tens__o__kV_"] < 750)]
            lt_345   = gdf_lt[(gdf_lt["Tens__o__kV_"] >= 345) & (gdf_lt["Tens__o__kV_"] < 500)]
            lt_230   = gdf_lt[(gdf_lt["Tens__o__kV_"] >= 230) & (gdf_lt["Tens__o__kV_"] < 345)]
            lt_baixa = gdf_lt[gdf_lt["Tens__o__kV_"] < 230]

            if len(lt_750)   > 0: lt_750.plot(ax=ax,   color="#7b0000", linewidth=2.5, alpha=0.9)
            if len(lt_500)   > 0: lt_500.plot(ax=ax,   color="#d62728", linewidth=1.8, alpha=0.8)
            if len(lt_345)   > 0: lt_345.plot(ax=ax,   color="#ff7f0e", linewidth=1.3, alpha=0.7)
            if len(lt_230)   > 0: lt_230.plot(ax=ax,   color="#1f77b4", linewidth=1.0, alpha=0.6)
            if len(lt_baixa) > 0: lt_baixa.plot(ax=ax, color="#aec7e8", linewidth=0.7, alpha=0.5)

            cores_risco = {"🔴 CRÍTICO": ("red", 100), "🟠 ALTO": ("orange", 80), "🟡 MÉDIO": ("yellow", 60), "🟢 BAIXO": ("lime", 50)}
            for nivel, (cor, tam) in cores_risco.items():
                subset = gdf_focos_proj[gdf_focos_proj["risco"] == nivel]
                if len(subset) > 0:
                    gpd.GeoDataFrame(subset, geometry="geometry", crs="EPSG:31983").to_crs("EPSG:4326").plot(
                        ax=ax, color=cor, markersize=tam, alpha=0.9, marker="o", zorder=5)

            legenda = [
                mlines.Line2D([], [], color="#7b0000", linewidth=3, label=f"LT ≥ 750 kV ({len(lt_750)})"),
                mlines.Line2D([], [], color="#d62728", linewidth=3, label=f"LT 500 kV ({len(lt_500)})"),
                mlines.Line2D([], [], color="#ff7f0e", linewidth=3, label=f"LT 345 kV ({len(lt_345)})"),
                mlines.Line2D([], [], color="#1f77b4", linewidth=3, label=f"LT 230 kV ({len(lt_230)})"),
                mlines.Line2D([], [], color="#aec7e8", linewidth=3, label=f"LT < 230 kV ({len(lt_baixa)})"),
                mlines.Line2D([], [], color="red",    marker="o", markersize=10, linestyle="None", label="🔴 Crítico (≤ 10 km)"),
                mlines.Line2D([], [], color="orange", marker="o", markersize=10, linestyle="None", label="🟠 Alto (≤ 50 km)"),
                mlines.Line2D([], [], color="yellow", marker="o", markersize=10, linestyle="None", label="🟡 Médio (≤ 100 km)"),
                mlines.Line2D([], [], color="lime",   marker="o", markersize=10, linestyle="None", label="🟢 Baixo (> 100 km)"),
            ]
            ax.legend(handles=legenda, fontsize=9, loc="lower left", framealpha=0.9)
            ax.set_title(f"Score de Risco Climático — {datetime.now().strftime('%d/%m/%Y %H:%M')}", fontsize=13)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("📋 Tabela de Alertas")
            cols = ["lat", "lon", "dist_km", "linha_nome", "tensao_kv", "sub_origem", "sub_destino", "risco"]
            df_alertas = gdf_focos_proj[cols].sort_values("dist_km").head(20).round(2)
            df_alertas.columns = ["Lat", "Lon", "Dist (km)", "Linha", "Tensão (kV)", "Sub Origem", "Sub Destino", "Risco"]
            st.dataframe(df_alertas, use_container_width=True)

# ============================================================
# PÁGINA 3 — GERAÇÃO ONS
# ============================================================
elif pagina == "⚡ Geração ONS":
    st.title("⚡ Análise de Geração — ONS")
    st.caption("Fator de capacidade de usinas solares e eólicas")
    st.divider()

    df_ons = carregar_ons()
    if df_ons is None:
        st.error("Arquivo ONS não encontrado em data/raw/")
    else:
        df_solar  = df_ons[df_ons["nom_tipousina"] == "Solar"]
        df_eolica = df_ons[df_ons["nom_tipousina"] == "Eólica"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Fator Médio Solar", f"{df_solar['val_fatorcapacidade'].mean():.3f}")
        with col2:
            st.metric("Fator Médio Eólico", f"{df_eolica['val_fatorcapacidade'].mean():.3f}")
        with col3:
            st.metric("Melhor Subsistema Solar", df_solar.groupby("nom_subsistema")["val_fatorcapacidade"].mean().idxmax())
        with col4:
            st.metric("Melhor Subsistema Eólico", df_eolica.groupby("nom_subsistema")["val_fatorcapacidade"].mean().idxmax())

        st.subheader("Fator de Capacidade por Subsistema")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df_solar.groupby("nom_subsistema")["val_fatorcapacidade"].mean().plot(kind="bar", ax=axes[0], color="#f4c430")
        axes[0].set_title("Solar")
        axes[0].set_xlabel("Subsistema")
        axes[0].set_ylabel("Fator médio")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True, alpha=0.3)

        df_eolica.groupby("nom_subsistema")["val_fatorcapacidade"].mean().plot(kind="bar", ax=axes[1], color="#4a90d9")
        axes[1].set_title("Eólica")
        axes[1].set_xlabel("Subsistema")
        axes[1].set_ylabel("Fator médio")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Sazonalidade — Fator de Capacidade por Hora")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        solar_hora = df_solar.groupby("hora")["val_fatorcapacidade"].mean()
        axes[0].plot(solar_hora.index, solar_hora.values, color="#f4c430", linewidth=2.5, marker="o", markersize=4)
        axes[0].fill_between(solar_hora.index, solar_hora.values, alpha=0.3, color="#f4c430")
        axes[0].set_title("Solar — Fator por Hora")
        axes[0].set_xlabel("Hora do Dia")
        axes[0].set_ylabel("Fator médio")
        axes[0].set_xticks(range(0, 24, 2))
        axes[0].grid(True, alpha=0.3)

        eolica_hora = df_eolica.groupby("hora")["val_fatorcapacidade"].mean()
        axes[1].plot(eolica_hora.index, eolica_hora.values, color="#4a90d9", linewidth=2.5, marker="o", markersize=4)
        axes[1].fill_between(eolica_hora.index, eolica_hora.values, alpha=0.3, color="#4a90d9")
        axes[1].set_title("Eólica — Fator por Hora")
        axes[1].set_xlabel("Hora do Dia")
        axes[1].set_ylabel("Fator médio")
        axes[1].set_xticks(range(0, 24, 2))
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

# ============================================================
# PÁGINA 4 — PREVISÃO ML
# ============================================================
elif pagina == "🤖 Previsão ML":
    st.title("🤖 Previsão de Geração — Machine Learning")
    st.caption("Modelo XGBoost treinado com dados ONS, NASA POWER e Copernicus ERA5")
    st.divider()

    modelo_solar, modelo_eolico = carregar_modelos()

    if modelo_solar is None:
        st.error("Modelos não encontrados. Execute o Notebook 03 primeiro.")
    else:
        st.success("✅ Modelos carregados | Solar R²=0.880 | Eólico R²=0.723")

        st.subheader("🎯 Simular Previsão")
        col1, col2, col3 = st.columns(3)

        with col1:
            hora       = st.slider("Hora do dia", 0, 23, 12)
            mes        = st.slider("Mês", 1, 12, 6)
            dia_semana = st.slider("Dia da semana (0=Seg)", 0, 6, 1)

        with col2:
            irradiancia = st.slider("Irradiância (W/m²)", 0, 1000, 500)
            temperatura = st.slider("Temperatura (°C)", 10, 45, 25)
            nuvens      = st.slider("Cobertura de nuvens (%)", 0, 100, 30)

        with col3:
            vento_10m  = st.slider("Vento 10m (m/s)", 0.0, 15.0, 5.0)
            vento_100m = st.slider("Vento 100m (m/s)", 0.0, 20.0, 7.0)
            capacidade = st.number_input("Capacidade instalada (MW)", 10, 500, 100)

        if st.button("🔮 Gerar Previsão", type="primary"):
            hora_sin = np.sin(2 * np.pi * hora / 24)
            hora_cos = np.cos(2 * np.pi * hora / 24)
            mes_sin  = np.sin(2 * np.pi * mes / 12)
            mes_cos  = np.cos(2 * np.pi * mes / 12)
            dia_sin  = np.sin(2 * np.pi * (mes * 30) / 365)
            dia_cos  = np.cos(2 * np.pi * (mes * 30) / 365)

            X = pd.DataFrame([[
                hora, mes, dia_semana, (mes-1)//3 + 1,
                hora_sin, hora_cos, mes_sin, mes_cos, dia_sin, dia_cos,
                -10.0, -50.0, capacidade, 1, 5,
                irradiancia, temperatura, vento_10m, nuvens,
                vento_10m, vento_100m, irradiancia / 3600
            ]], columns=[
                "hora", "mes", "dia_semana", "trimestre",
                "hora_sin", "hora_cos", "mes_sin", "mes_cos", "dia_ano_sin", "dia_ano_cos",
                "latitude", "longitude", "capacidade", "subsistema_enc", "estado_enc",
                "irradiancia_nasa", "temperatura_c", "vento_nasa_ms", "nuvens_pct",
                "vento_10m_ms", "vento_100m_ms", "ssrd_wm2"
            ])

            fc_solar  = float(modelo_solar.predict(X)[0])
            fc_eolico = float(modelo_eolico.predict(X)[0])
            fc_solar  = max(0, min(1, fc_solar))
            fc_eolico = max(0, min(1, fc_eolico))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("☀️ Fator de Capacidade Solar", f"{fc_solar:.3f}", f"{fc_solar*capacidade:.1f} MWh estimado")
            with col2:
                st.metric("💨 Fator de Capacidade Eólico", f"{fc_eolico:.3f}", f"{fc_eolico*capacidade:.1f} MWh estimado")

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].bar(["Solar"], [fc_solar], color="#f4c430", edgecolor="black")
            axes[0].set_ylim(0, 1)
            axes[0].set_title(f"☀️ Fator Solar: {fc_solar:.3f}")
            axes[0].set_ylabel("Fator de Capacidade")
            axes[0].grid(True, alpha=0.3, axis="y")
            axes[0].axhline(0.2, color="red", linestyle="--", alpha=0.5, label="Média histórica")
            axes[0].legend()

            axes[1].bar(["Eólica"], [fc_eolico], color="#4a90d9", edgecolor="black")
            axes[1].set_ylim(0, 1)
            axes[1].set_title(f"💨 Fator Eólico: {fc_eolico:.3f}")
            axes[1].set_ylabel("Fator de Capacidade")
            axes[1].grid(True, alpha=0.3, axis="y")
            axes[1].axhline(0.26, color="red", linestyle="--", alpha=0.5, label="Média histórica")
            axes[1].legend()

            plt.tight_layout()
            st.pyplot(fig)
