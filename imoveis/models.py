from config import db, login_manager
from flask_login import UserMixin

# o ID do usuário é salvo na sessão quando ele faz login
# a cada nova requisição, load_user(user_id) busca o usuário no banco de dados com esse ID
# isso permite ao Flask-Login gerenciar o estado de login e garantir que o usuário correto esteja autenticado
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# essa classe representará uma tabela chamada user no banco de dados
class User(db.Model, UserMixin): # db.Model é uma classe base do SQLAlchemy, que indica que a classe User será mapeada para uma tabela no banco de dados
    # cada atributo da classe User é uma coluna na tabela do banco de dados
    id = db.Column(db.Integer, primary_key=True) # coluna id
    username = db.Column(db.String(150), unique=True, nullable=False) # coluna nome de usuário
    password = db.Column(db.String(150), nullable=False) # coluna senha
    