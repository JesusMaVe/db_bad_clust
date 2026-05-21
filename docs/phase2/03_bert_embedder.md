# Módulo 3: BERT Embedder

## Propósito

Genera vectores densos ℝ⁷⁶⁸ para cada atributo de base de datos usando el modelo `bert-base-multilingual-cased` de Hugging Face. Estos embeddings capturan relaciones semánticas entre nombres de columna (ej. `customer_name` ≈ `nombre_cliente`).

## Ubicación

`scripts/bert_embedder.py`

## Dependencias

- `transformers>=4.30.0`
- `torch>=2.0.0`
- `numpy`

Instalar:

```bash
pip install transformers torch numpy
```

## API

```python
from bert_embedder import BERTEmbedder

# Inicializar (carga el modelo lazy — solo cuando se llama a encode)
embedder = BERTEmbedder(
    model_name="bert-base-multilingual-cased",
    device="auto",        # "auto" | "cpu" | "cuda" | "mps"
    max_length=32,        # tokens máximos por texto
)

# Batch encoding
texts = ["empleados: fecha nacimiento", "clientes: identifier cliente"]
vectors = embedder.encode(texts)  # → numpy.ndarray shape (N, 768)

# Single
vec = embedder.encode_single("empleados: fecha nacimiento")  # → shape (768,)

# Liberar memoria
embedder.unload()
```

## Pipeline Interno

```
Textos preprocesados
  → Tokenizer (padding, truncation, max_length=32)
  → BERT model (forward pass, no grad)
  → Extraer [CLS] vector (última capa, primer token)
  → numpy array float32
```

## Modelo

| Propiedad         | Valor                         |
| ----------------- | ----------------------------- |
| Modelo            | `bert-base-multilingual-cased` |
| Dimensión         | 768                           |
| Capas             | 12                            |
| Cabezas           | 12                            |
| Parámetros        | ~177M                         |
| Idiomas           | 104 (incluye español)         |

## Notas

- El modelo se **cachea** en `~/.cache/huggingface/` después de la primera descarga (~680MB).
- Usa el vector del token `[CLS]` como representación del texto completo.
- `device="auto"` detecta CUDA → MPS → CPU en ese orden.
- La primera llamada a `encode()` descarga los pesos si no están cacheados.
