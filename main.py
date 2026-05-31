from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from typing import List
import uuid
import os
import platform
import traceback
import anyio

from training import training as _training
from training.training import train_model, _load_cache_if_needed
from training.xgboost_training import train_xgb, load_xgb_model, recommend_with_xgb
from training.kmeans_training import load_kmeans_model, train_kmeans, KMEANS_MODEL_FILE
from training.knn_training import load_knn_model, train_knn, KNN_MODEL_FILE
from recommending.recommendation import get_hybrid_recommendations


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- INÍCIO LIFESPAN ---")
    print(f"Ambiente: OS={platform.system()}, Architecture={platform.machine()}")
    print(f"Variáveis de Ambiente (PGHOST, etc.): PGHOST={os.getenv('PGHOST', 'N/A')}")

    try:
        _load_cache_if_needed()
    except Exception as e:
        print(f"Aviso: falha ao carregar modelos cacheados (training.py): {e}")

    if _training._item_df is None:
        print("[lifespan] item_features.pkl não encontrado — treinando modelo Content-Based...")
        try:
            train_model()
        except Exception as e:
            print(f"Aviso: falha ao treinar modelo Content-Based: {e}")

    if not os.path.exists(KMEANS_MODEL_FILE):
        print("[lifespan] kmeans_model.pkl não encontrado — treinando K-Means...")
        try:
            train_kmeans()
        except Exception as e:
            print(f"Aviso: falha ao treinar K-Means: {e}")
    else:
        try:
            load_kmeans_model()
        except Exception as e:
            print(f"Aviso: falha ao carregar modelo K-Means: {e}")

    if not os.path.exists(KNN_MODEL_FILE):
        print("[lifespan] knn_model.pkl não encontrado — treinando KNN...")
        try:
            train_knn()
        except Exception as e:
            print(f"Aviso: falha ao treinar KNN: {e}")
    else:
        try:
            load_knn_model()
        except Exception as e:
            print(f"Aviso: falha ao carregar modelo KNN: {e}")

    yield
    print("Aplicação encerrada.")


app = FastAPI(
    lifespan=lifespan,
    title="Serviço de Recomendação de Joias",
    description="Uma API para fornecer recomendações de joias baseadas em conteúdo e interações.",
    version="1.1.0"
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_details = exc.errors()
    print("\n--- ERRO DE VALIDAÇÃO (422 UNPROCESSABLE CONTENT) ---")
    print(f"URL: {request.url}")
    for error in error_details:
        print(f"  - Campo: {error['loc']}, Mensagem: {error['msg']}")
    print("--------------------------------------------------\n")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": error_details},
    )


@app.get("/diagnostic")
def diagnostic_data():
    item_count = len(_training._item_df) if _training._item_df is not None else 0
    pop_count = len(_training._popularity) if _training._popularity is not None else 0

    xgb_status    = "Carregado OK" if load_xgb_model()    is not None else "N/A (não encontrado)"
    kmeans_status = "Carregado OK" if load_kmeans_model() is not None else "N/A (não encontrado)"
    knn_status    = "Carregado OK" if load_knn_model()    is not None else "N/A (não encontrado)"

    if item_count == 0 or pop_count == 0:
        return {
            "status": "CRÍTICO: Dados de treinamento ausentes",
            "item_count": item_count,
            "popularity_count": pop_count,
            "xgb_model_status": xgb_status,
            "kmeans_model_status": kmeans_status,
            "knn_model_status": knn_status,
        }

    return {
        "status": "OK: Dados Carregados",
        "item_count": item_count,
        "popularity_count": pop_count,
        "first_popular_item": _training._popularity[0] if pop_count > 0 else None,
        "xgb_model_status": xgb_status,
        "kmeans_model_status": kmeans_status,
        "knn_model_status": knn_status,
    }


class RecommendationRequest(BaseModel):
    userId: str = Field(alias='user_id')
    numberOfRecommendations: int = Field(alias='number_of_recommendations')
    model_config = ConfigDict(populate_by_name=True)


class RecommendationResponse(BaseModel):
    recommendedForYou: List[str]
    popularNow: List[str]


@app.get("/")
def read_root():
    return {"status": "Serviço de ML online"}


@app.post("/echo")
async def echo():
    return {"ok": True}


@app.post("/ping", response_model=RecommendationResponse)
async def recommend_ping(request: RecommendationRequest):
    """Endpoint de diagnóstico: retorna popularidade sem DB ou ML."""
    pop = [str(x) for x in (_training._popularity or [])[:request.numberOfRecommendations]]
    return RecommendationResponse(recommendedForYou=pop, popularNow=pop)


async def _run_recommendations(user_id_str: str, count: int) -> RecommendationResponse:
    fallback = [str(x) for x in (_training._popularity or [])[:count]]
    recs_gosto: List[str] = fallback
    recs_xgb: List[str] = fallback

    print(f"[/recommendations] Iniciando para user_id={user_id_str}, count={count}")

    try:
        with anyio.fail_after(20):
            print("[/recommendations] Chamando get_hybrid_recommendations...")
            recs_gosto = await anyio.to_thread.run_sync(
                lambda: get_hybrid_recommendations(user_id=user_id_str, count=count)
            )
            print(f"[/recommendations] get_hybrid_recommendations OK ({len(recs_gosto)} itens)")

            print("[/recommendations] Chamando recommend_with_xgb...")
            recs_xgb = await anyio.to_thread.run_sync(
                lambda: recommend_with_xgb(user_id=user_id_str, count=count)
            )
            print(f"[/recommendations] recommend_with_xgb OK ({len(recs_xgb)} itens)")

    except TimeoutError:
        print(f"[/recommendations] TIMEOUT (>20s) para user_id={user_id_str} — retornando fallback")
    except Exception as e:
        print(f"[/recommendations] ERRO: {e}")
        traceback.print_exc()

    return RecommendationResponse(
        recommendedForYou=[str(r) for r in recs_gosto],
        popularNow=[str(r) for r in recs_xgb]
    )


@app.post("/recommendations", response_model=RecommendationResponse)
async def recommend_post(request: RecommendationRequest):
    return await _run_recommendations(str(request.userId), int(request.numberOfRecommendations))


@app.get("/recommendations", response_model=RecommendationResponse)
async def recommend_get(user_id: str, number_of_recommendations: int = 10):
    return await _run_recommendations(user_id, number_of_recommendations)
