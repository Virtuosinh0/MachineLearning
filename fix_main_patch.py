from pathlib import Path

p = Path('main.py')
text = p.read_bytes().decode('utf-8', errors='replace')
old = '        return RecommendationResponse(recommendedForYou=[], popularNow=[])'
new = '        raise HTTPException(\n            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,\n            detail="Falha ao processar a recomendação. Verifique os logs do serviço."\n        )'
if old not in text:
    raise RuntimeError('Old string not found')
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print('Patch applied')
