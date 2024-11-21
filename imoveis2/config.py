from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__) # criando um site vazio

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///usuarios.db'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # limita a 16MB o tamanho do arquivo
app.secret_key = "your_secret_key"  # necessário para usar sessões

# SECRET_KEY -> é usado para proteger dados sensíveis, ele garante que os dados transmitidos entre o servidor e o navegador sejam criptografados
# SQLALCHEMY_DATABASE_URI -> define qual banco de dados sua aplicação vai usar e onde ele está localizado

db = SQLAlchemy() # cria a instância do SQLAlchemy sem vinculá-la imediatamente à aplicação
db.init_app(app) # inicializa a aplicação com o SQLAlchemy

login_manager = LoginManager() # cria uma instância de LoginManager, que é responsável por gerenciar tudo relacionado à autenticação de usuários na aplicação
login_manager.init_app(app) # liga o LoginManager à aplicação Flask
login_manager.login_view = 'login' # define a rota que será usada quando um usuário tentar acessar uma página que requer autenticação, mas não estiver logado
