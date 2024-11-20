from config import app, db
from models import *

# código para criar o banco de dados
# precisa de um contexto
try:
    with app.app_context():
        db.create_all() # cria as tabelas no banco de dados se elas não existirem
    print("Banco de dados criado com sucesso!")
except Exception as e:
    print(f"Erro ao criar o banco: {e}")
    