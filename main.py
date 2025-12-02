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
import traceback

from training.training import train_model, _load_cache_if_needed, _item_df, _popularity
from training.xgboost_training import train_xgb
from training.xgboost_training import load_xgb_model
from recommending.recommendation import get_recommendations
from training.xgboost_training import recommend_with_xgb

async def lifespan(app: FastAPI):
    print(f"--- INÍCIO LIFESPAN ---")
    print(f"Ambiente: OS={platform.system()}, Architecture={platform.machine()}")
    print(f"Variáveis de Ambiente (PGHOST, etc.): PGHOST={os.getenv('PGHOST', 'N/A')}")
    
    print("ATENÇÃO: Treinamento desabilitado. Carregando modelos em cache (PKL) do disco...")

    try:
        _load_cache_if_needed() 
    except Exception as e:
        print(f"Aviso: falha ao carregar modelos cacheados (training.py): {e}")

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
    
    xgb_status = "N/A (Falha ao carregar)"
    xgb_model = load_xgb_model()
    if xgb_model is not None:
        xgb_status = "Carregado OK"
    else:
        print("[diagnostic] Tentativa de carregamento de XGBoost na requisição...")
        load_xgb_model() 
    # ---------------------------------------------

    if item_count == 0 or pop_count == 0:
        return {
            "status": "CRÍTICO: Dados de treinamento ausentes",
            "item_count": item_count,
            "popularity_count": pop_count,
            "xgb_model_status": xgb_status, # Novo campo
            "detalhes": "Se 'item_count' for 0, o problema está na consulta SQL à tabela 'jewelries'. Se 'popularity_count' for 0, o problema está na tabela 'user_interaction'. Verifique o conteúdo do seu banco de dados no Railway."
        }
    
    return {
        "status": "OK: Dados Carregados",
        "item_count": item_count,
        "popularity_count": pop_count,
        "first_popular_item": _popularity[0] if pop_count > 0 else None,
        "xgb_model_status": xgb_status # Novo campo
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
    
@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest): 
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
        print(f"--- ERRO CRÍTICO NO ENDPOINT /RECOMMENDATIONS ---") 
        print(f"Erro inesperado ao processar a requisição: {e}") 
        traceback.print_exc() 
        print("-------------------------------------------------") 
        return RecommendationResponse(recommendedForYou=[], popularNow=[])