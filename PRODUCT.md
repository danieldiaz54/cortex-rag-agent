# PRODUCT.md — `cortex-rag-agent`

## 1. Propósito en una frase
**Chat inteligente con tus documentos**: sube manuales, contratos o apuntes y haz preguntas normales obteniendo respuestas claras con enlace directo al párrafo exacto que las respalda.

## 2. Regla de 2 Capas (Usuario primero, Arquitectura bajo demanda)
- **Capa 1 (`Read + Operate`):** Estudio dividido en 2 paneles: conversación a la izquierda (pregunta en lenguaje natural + 3 preguntas sugeridas clicables + chips de cita `[1]`) y lector limpio del documento fuente a la derecha que resalta al instante el fragmento citado en ámbar cálido al hacer clic. Permite subir nuevos documentos (`+ Agregar documento`).
- **Capa 2 (Cajón `"Arquitectura"` opcional):** Muestra puntajes combinados de búsqueda híbrida (BM25 + Vectores 64-D RRF), guardrail anti-alucinaciones, ejecución de suite de evaluación (`POST /api/v1/evals/run`) y registro inmutable (`GET /api/v1/evals/ledger`).
