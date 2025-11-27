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

from training.training import train_model
from training.xgboost_training import train_xgb
from recommending.recommendation import get_recommendations
from training.xgboost_training import recommend_with_xgb

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Iniciando a aplicação e treinando o modelo...")
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