import streamlit as st
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def load_data():
    """
    Inicializa la sesión Spark y carga el archivo CSV.
    Asegúrate de tener el archivo "data.csv" en el directorio del proyecto.
    """
    spark = SparkSession.builder.appName("FRC2025Reefscape").getOrCreate()
    df = spark.read.option("header", True).csv("data.csv")
    return spark, df

def transform_data(df):
    """
    Transforma la estructura original para que cada fila represente un equipo en un match.
    Se separa la información de la alianza roja y azul, renombrando las columnas para homogeneizarlas.
    """
    red_positions = ["red1", "red2", "red3"]
    blue_positions = ["blue1", "blue2", "blue3"]

    red_dfs = []
    for pos in red_positions:
        red_df = df.select(
            F.col("key"),
            F.col(f"{pos}_team").alias("team"),
            F.col(f"{pos}_epa").alias("epa"),
            F.col(f"{pos}_total_points").alias("total_points"),
            F.col(f"{pos}_auto_points").alias("auto_points"),
            F.col(f"{pos}_teleop_points").alias("teleop_points"),
            F.col(f"{pos}_endgame_points").alias("endgame_points"),
            F.col(f"{pos}_winrate").alias("winrate"),
            F.col(f"{pos}_rank").alias("rank")
        ).withColumn("alliance", F.lit("red"))
        red_dfs.append(red_df)
    
    red_all = red_dfs[0].union(red_dfs[1]).union(red_dfs[2])
    
    blue_dfs = []
    for pos in blue_positions:
        blue_df = df.select(
            F.col("key"),
            F.col(f"{pos}_team").alias("team"),
            F.col(f"{pos}_epa").alias("epa"),
            F.col(f"{pos}_total_points").alias("total_points"),
            F.col(f"{pos}_auto_points").alias("auto_points"),
            F.col(f"{pos}_teleop_points").alias("teleop_points"),
            F.col(f"{pos}_endgame_points").alias("endgame_points"),
            F.col(f"{pos}_winrate").alias("winrate"),
            F.col(f"{pos}_rank").alias("rank")
        ).withColumn("alliance", F.lit("blue"))
        blue_dfs.append(blue_df)
    
    blue_all = blue_dfs[0].union(blue_dfs[1]).union(blue_dfs[2])
    full_df = red_all.union(blue_all)
    
    # Conversión de columnas numéricas a float
    numeric_cols = ["epa", "total_points", "auto_points", "teleop_points", "endgame_points", "winrate", "rank"]
    for col in numeric_cols:
        full_df = full_df.withColumn(col, F.col(col).cast("float"))
    
    return full_df

def aggregate_metrics(df):
    """
    Agrega métricas por equipo y alianza: número de partidos jugados, promedios de puntos y otros.
    """
    agg_df = df.groupBy("team", "alliance").agg(
        F.count("*").alias("matches_played"),
        F.avg("total_points").alias("avg_total_points"),
        F.avg("epa").alias("avg_epa"),
        F.avg("winrate").alias("avg_winrate"),
        F.avg("auto_points").alias("avg_auto_points"),
        F.avg("teleop_points").alias("avg_teleop_points"),
        F.avg("endgame_points").alias("avg_endgame_points")
    )
    return agg_df

def main():
    st.set_page_config(page_title="Dashboard FRC 2025 Reefscape", layout="wide")
    st.title("Dashboard FRC 2025 Reefscape")
    st.markdown(
        """
        Este dashboard muestra un análisis interactivo de los más de 8.000 matches y 2.000 equipos de la competencia.
        Utiliza filtros en la barra lateral para explorar datos por alianza, equipos y número mínimo de partidos jugados.
        """
    )

    # Cargar y transformar los datos
    spark, df_raw = load_data()
    st.info("Cargando y transformando datos...")
    df_transformed = transform_data(df_raw)
    df_aggregated = aggregate_metrics(df_transformed)
    
    # Convertir a Pandas para visualización
    df_pandas = df_aggregated.toPandas()
    
    # Barra lateral con filtros interactivos
    st.sidebar.header("Filtros de Datos")
    # Filtrado por alianza
    alliance_options = ["red", "blue"]
    alliance_filter = st.sidebar.multiselect("Selecciona alianza(s):", options=alliance_options, default=alliance_options)
    
    # Filtrado por equipos (se obtienen de los datos)
    teams = sorted(df_pandas["team"].unique())
    team_filter = st.sidebar.multiselect("Selecciona equipo(s):", options=teams)
    
    # Filtrado por mínimo de matches jugados
    min_matches = st.sidebar.slider("Mínimo de matches jugados:", min_value=0, max_value=int(df_pandas["matches_played"].max()), value=0)
    
    # Aplicar filtros
    filtered_df = df_pandas[
        (df_pandas["alliance"].isin(alliance_filter)) &
        (df_pandas["team"].isin(team_filter)) &
        (df_pandas["matches_played"] >= min_matches)
    ]
    
    st.subheader("Métricas Agregadas por Equipo")
    st.dataframe(filtered_df.sort_values(by="avg_total_points", ascending=False))
    
    # Selección de métrica para visualizar
    metric_options = {
        "Total de Puntos": "avg_total_points",
        "EPA": "avg_epa",
        "Winrate": "avg_winrate",
        "Auto Points": "avg_auto_points",
        "Teleop Points": "avg_teleop_points",
        "Endgame Points": "avg_endgame_points"
    }
    metric_label = st.selectbox("Selecciona la métrica a visualizar:", options=list(metric_options.keys()))
    metric = metric_options[metric_label]
    
    st.subheader(f"Gráfico de {metric_label} por Equipo")
    # Configurar el gráfico de barras
    chart_data = filtered_df.set_index("team")[metric].sort_values(ascending=False)
    st.bar_chart(chart_data)
    
    # Opción para mostrar los datos filtrados
    if st.checkbox("Mostrar datos agregados"):
        st.subheader("Datos Agregados Filtrados")
        st.write(filtered_df)

if __name__ == "__main__":
    main()
