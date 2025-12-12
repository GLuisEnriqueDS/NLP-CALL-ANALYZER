import pandas as pd
from datetime import datetime
from tqdm import tqdm

from utils.preprocessing import clean_text, lemmatize_list
from utils.clustering import generate_embeddings, cluster_embeddings, top_n_representatives
from utils.llm_analysis import analyze_cluster

print("="*60)
print("     CLUSTERING DE TRANSCRIPCIONES (Pipeline Limpio)")
print("="*60)

tqdm.pandas()

# === 1. Cargar datos ===
data = pd.read_excel("data/sample_data.xlsx")

def main():
    df = data.copy()

    print(f"Datos cargados: {len(df)} registros")

    # === 2. Limpieza ===
    df["limpio"] = df["transcripcion"].progress_apply(clean_text)

    # === 3. Lemmatización ===
    df["lemma"] = lemmatize_list(df["limpio"].tolist())

    # === 4. Embeddings ===
    df["embedding"] = generate_embeddings(df["lemma"].tolist()).tolist()

    # === 5. Clustering ===
    df, kmeans = cluster_embeddings(df, k=5)

    # === 6. Análisis de clusters ===
    results = {}
    df["representativa"] = 0

    for cl in sorted(df["cluster"].unique()):
        print(f"\n Analizando cluster {cl} ...") 

        subset = df[df["cluster"] == cl]

        idxs, reps = top_n_representatives(subset, kmeans, cl, n=1)

        df.loc[idxs, "representativa"] = 1
        results[cl] = analyze_cluster(reps, cl)

        print(f"Cluster {cl} procesado (docs: {len(subset)})") 


    # === 7. Mapear resultados al dataframe ===
    df["pregunta_representativa"] = df["cluster"].map(lambda c: results[c]["pregunta_representativa"])
    df["categorias"] = df["cluster"].map(lambda c: results[c]["categorias"])
    df["cluster_name"] = df["cluster"].map(lambda c: results[c]["cluster_name"])

    # == 8.  Resumen por cluster
    clusters = sorted(df["cluster"].unique())

    for cl in clusters:
        subset = df[df["cluster"] == cl]

        print(f"\n📌 Cluster {cl} — {results[cl]['cluster_name']}")
        print(f"   • Documentos: {len(subset)}")
        print(f"   • Pregunta representativa: {results[cl]['pregunta_representativa'][:100]}...")
        print(f"   • Categorías: {', '.join(results[cl]['categorias'])}")

    # === 9. Guardar output ===
    fname = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(fname, index=False, sep=";", encoding="utf-8-sig")

    print("\nProceso completado. Archivo guardado:", fname)

if __name__ == "__main__":
    main()