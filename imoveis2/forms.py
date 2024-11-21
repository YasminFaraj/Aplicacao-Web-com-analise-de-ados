from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length

# criamos uma classe RegisterForm que será usada para criar um formulário de registro de usuário
class RegisterForm(FlaskForm):
    # criando os campos do formulário
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Registrar')

# criamos uma classe LoginForm que será usada para criar um formulário de login de usuário
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')
    