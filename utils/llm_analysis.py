import json
import ollama

def analyze_cluster(transcriptions, cl):
    textos = "\n---\n".join([t[:800] for t in transcriptions])

    prompt = f"""
    Analiza las siguientes transcripciones representativas de un cluster...

    TRANSCRIPCIONES:
    {textos}

    FORMATO JSON:
    {{
        "pregunta_representativa": "...",
        "categorias": ["cat1"],
        "cluster_name": "Nombre Corto"
    }}
    """

    r = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )

    raw = r["message"]["content"]

    try:
        json_str = raw[raw.find("{"): raw.rfind("}") + 1]
        return json.loads(json_str)
    except:
        return {
            "pregunta_representativa": transcriptions[0][:150],
            "categorias": ["general"],
            "cluster_name": f"cluster_{cl}"
        }
