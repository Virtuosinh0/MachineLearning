from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException, Request, status 
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
import uuid
from pydantic.config import ConfigDict
import os
import platform

from training.training import train_model, _load_cache_if_needed, _item_df, _popularity
from training.xgboost_training import train_xgb
from recommending.recommendation import get_recommendations
from training.xgboost_training import recommend_with_xgb

async def lifespan(app: FastAPI):
    print(f"--- INÍCIO LIFESPAN ---")
    print(f"Ambiente: OS={platform.system()}, Architecture={platform.machine()}")
    print(f"Variáveis de Ambiente (PGHOST, etc.): PGHOST={os.getenv('PGHOST', 'N/A')}")
    
    try:
        train_model(limit_days=365) 
        train_xgb(limit_days=365, neg_ratio=3)
    except Exception as e:
        print(f"Aviso: falha ao treinar o modelo na inicialização: {e}")
        print("A aplicação continuará, tentando carregar modelos cacheados (se existirem).")

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
    print("Detalhes do Erro:")
    for error in error_details:
        print(f"  - Campo: {error['loc']}, Mensagem: {error['msg']}")
    print("--------------------------------------------------\n")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": error_details},
    )
    
@app.get("/diagnostic")
def diagnostic_data():
    item_count = len(_item_df) if _item_df is not None else 0
    pop_count = len(_popularity) if _popularity is not None else 0
    
    if item_count == 0 or pop_count == 0:
        return {
            "status": "CRÍTICO: Dados de treinamento ausentes",
            "item_count": item_count,
            "popularity_count": pop_count,
            "detalhes": "Se 'item_count' for 0, o problema está na consulta SQL à tabela 'jewelries'. Se 'popularity_count' for 0, o problema está na tabela 'user_interaction'. Verifique o conteúdo do seu banco de dados no Railway."
        }
    
    return {
        "status": "OK: Dados Carregados",
        "item_count": item_count,
        "popularity_count": pop_count,
        "first_popular_item": _popularity[0] if pop_count > 0 else None
    }


class RecommendationRequest(BaseModel):
    userId: uuid.UUID = Field(alias='user_id') 
    numberOfRecommendations: int = Field(alias='number_of_recommendations')
    model_config = ConfigDict(populate_by_name=True)

class RecommendationResponse(BaseModel):
    recommendedForYou: List[uuid.UUID]
    popularNow: List[uuid.UUID]

@app.get("/")
def read_root():
    return {"status": "Serviço de ML online"}

@app.post("/recommendationsOld", response_model=RecommendationResponse)
def recommend_old(request: RecommendationRequest):
    try:
        user_id_str = str(request.userId)
        count = int(request.numberOfRecommendations)

        recs_gosto = get_recommendations(user_id=user_id_str, count=count)

        recs_xgb = recommend_with_xgb(user_id=user_id_str, count=count)

        def parse_uuids(id_list):
            valid_uuids = []
            for r in id_list:
                try:
                    valid_uuids.append(uuid.UUID(r))
                except Exception:
                    continue
            return valid_uuids
        
        return RecommendationResponse(
            recommendedForYou=parse_uuids(recs_gosto),
            popularNow=parse_uuids(recs_xgb)
        )
    except Exception as e:
        print(f"Erro ao processar a recomendação para o usuário {request.userId}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/recommendations", response_model=RecommendationResponse)
def recommend_new(request: RecommendationRequest): 
    try:
        user_id_str = str(request.userId)
        count = int(request.numberOfRecommendations)

        recs_gosto = get_recommendations(user_id=user_id_str, count=count)

        recs_xgb = recommend_with_xgb(user_id=user_id_str, count=count)

        def parse_uuids(id_list):
            valid_uuids = []
            for r in id_list:
                try:
                    valid_uuids.append(uuid.UUID(r))
                except Exception:
                    continue
            return valid_uuids
        
        return RecommendationResponse(
            recommendedForYou=parse_uuids(recs_gosto),
            popularNow=parse_uuids(recs_xgb)
        )
    except Exception as e:
        print(f"Erro inesperado no endpoint /recommendations: {e}") 
        raise HTTPException(status_code=500, detail=str(e))