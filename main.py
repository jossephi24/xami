from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np

#Inicializa la aplicación
app = FastAPI()

#Configura CORS para permitir comunicación con el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las solicitudes
    allow_methods=["*"],
    allow_headers=["*"],
)

#Carga el modelo preentrenado
modelo = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Modelo de datos que espera el endpoint
class TextoRequest(BaseModel):
    texto: str

# Función para dividir el texto en oraciones
def dividir_oraciones(texto: str) -> list:
    # Usa expresiones regulares para dividir por punto, signo de interrogación, exclamación y saltos de línea
    oraciones = re.split(r'(?<=[.!?]) +|\n', texto.strip())
    return [s for s in oraciones if s]

#Función para obtener embeddings de cada oración
def obtener_embeddings(oraciones: list) -> list:
    embeddings = modelo.encode(oraciones)
    print("Embeddings: ",embeddings,"\n")
    return embeddings

#Función para calcular similaridad coseno entre dos embeddings
def calcular_similaridad(embedding1, embedding2) -> float:
    # Reshape para asegurar forma (1, -1)
    print("embedding1 antes del reshape: ", embedding1, "\n");
    print("embedding2 antes del reshape: ", embedding2, "\n");
    
    emb1 = embedding1.reshape(1, -1)
    emb2 = embedding2.reshape(1, -1)
    similitud = cosine_similarity(emb1, emb2)[0][0]
    print("embedding1: ", embedding1, "\n");
    print("embedding2: ", embedding2, "\n");
    return similitud

#Función para interpretar el puntaje final de coherencia
def interpretar_coherencia(score: float) -> str:
    """
    Interpreta el puntaje de coherencia
        
    Args:
        score (float): Puntaje de coherencia
            
    Returns:
        str: Interpretación del puntaje
    """
    if score >= 0.8:
        return "Muy coherente"
    elif score >= 0.6:
        return "Coherente"
    elif score >= 0.4:
        return "Moderadamente coherente"
    elif score >= 0.2:
        return "Poco coherente"
    else:
        return "Muy poco coherente"


# Función para analizar coherencia entre oraciones
def analizar_coherencia(oraciones: list, embeddings: list, umbral: float = 0.6) -> dict:
    puntajes = []
    oraciones_no_coherentes = []
    
    print("Umbral: ", umbral, "\n")

    for i in range(len(oraciones) - 1):
        emb1 = embeddings[i]
        emb2 = embeddings[i + 1]
        similitud = calcular_similaridad(emb1, emb2)
        puntajes.append(similitud)

        # Si la similitud es menor al umbral, lo marcamos como incoherente
        #if similitud < umbral:
        oraciones_no_coherentes.append({
            "oracion_1": oraciones[i],
            "oracion_2": oraciones[i + 1],
            "coherencia": similitud  # en porcentaje
        })
    
    print("Similitudes consecutivas: ", puntajes, "\n")
    coherencia_consecutiva_promedio = float( sum(puntajes) / len(puntajes) ) if puntajes else 0

    # Calcular similitudes entre todas las oraciones
    matriz_similitudes = cosine_similarity(embeddings)
    print("Matriz Similitudes: ",matriz_similitudes, "\n")
    
    # Similitud promedio excluyendo la diagonal
    mascara = np.ones(matriz_similitudes.shape, dtype=bool)
    np.fill_diagonal(mascara, False)
    similitud_general_promedio = matriz_similitudes[mascara].mean()
    
    # Coherencia global (promedio de similitudes no diagonales)
    coherencia_global = similitud_general_promedio
    print("Coherencia Global: ", coherencia_global)

    # Coherencia local (similitudes consecutivas)
    coherencia_local = coherencia_consecutiva_promedio
    print("Coherencia Local: ", coherencia_local)
        
    # Puntaje final combinado
    coherencia_final = (coherencia_local * 0.6) + (coherencia_global * 0.4)

    intepretacion = interpretar_coherencia(coherencia_final)
    
    print("Coherencia final: ",coherencia_final, "\n")
    print("Interpretación: ",intepretacion, "\n")
    print("oraciones_no_coherentes: ",oraciones_no_coherentes,"\n")
    
    return {
        "coherencia_promedio": coherencia_final,
        "oraciones_no_coherentes": oraciones_no_coherentes,
        "intepretacion":intepretacion
    }

#Endpoint principal para analizar el texto
@app.post("/analizar")
async def analizar_texto(request: TextoRequest):
    texto = request.texto

    # Paso 1: Divide en oraciones
    oraciones = dividir_oraciones(texto)

    # Paso 2: Obtiene embeddings de cada oración
    embeddings = obtener_embeddings(oraciones)

    # Paso 3: Analiza coherencia
    resultado = analizar_coherencia(oraciones, embeddings, umbral=0.6)

    #return resultado
    return {
    "coherencia_promedio": float(resultado["coherencia_promedio"]),
    "intepretacion": (resultado["intepretacion"]),
    "oraciones_no_coherentes": [
        (x["oracion_1"], x["oracion_2"], float(x["coherencia"]))
        for x in resultado["oraciones_no_coherentes"]
    ]
}