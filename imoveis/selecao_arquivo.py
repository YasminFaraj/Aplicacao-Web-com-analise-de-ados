from flask import Flask, request, render_template, redirect, url_for, session
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import tempfile
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limita a 16MB o tamamho do arquivo
app.secret_key = "your_secret_key"  # Necessário para usar sessões

# Página inicial para upload do arquivo CSV
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session.clear()  # Limpa qualquer dado armazenado na sessão
        if 'arquivo' not in request.files:
            return "Nenhum arquivo foi enviado"

        arquivo = request.files['arquivo']

        if arquivo.filename == '':
            return "Escolha um arquivo válido"

        # Lê o arquivo CSV
        data = pd.read_csv(arquivo)

        # O arquivo estava grande, então essa parte cria um arquivo temporário para armazenas o csv
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as arq_temp:
            data.to_csv(arq_temp.name, index=False)
            session['arquivo_temporario'] = arq_temp.name  # Armazena o caminho do arquivo temporário na sessão

        # Redireciona para a página de seleção de colunas, passando as colunas do CSV
        return redirect(url_for('selecao_colunas'))

    return render_template('index.html')


# Página para seleção de colunas
@app.route('/selecao_colunas', methods=['GET', 'POST'])
def selecao_colunas():
    # Carrega o dataframe da sessão (os dados temporários)
    arq_temp_caminho = session.get('arquivo_temporario')
    if arq_temp_caminho is None:
        return redirect(url_for('index'))

    # Lendo o arquivo e reconhecendo cada coluna
    data = pd.read_csv(arq_temp_caminho)
    colunas = data.columns

    if request.method == 'POST':
        # Salva as colunas escolhidas pelo usuário
        session['tipo_col'] = request.form['tipo']
        session['cidade_col'] = request.form['cidade']
        session['bairro_col'] = request.form['bairro']
        session['metragem_col'] = request.form['metragem']
        session['quartos_col'] = request.form['quartos']
        session['banheiros_col'] = request.form['banheiros']
        session['preco_col'] = request.form['preco']
        session['latitude_col'] = request.form['latitude']
        session['longitude_col'] = request.form['longitude']

        # Redireciona para a página de configuração do modelo
        return redirect(url_for('configura_modelo'))

    return render_template('selecao_colunas.html', columns=colunas)


# Página para configuração do modelo e treinamento
@app.route('/configura_modelo', methods=['GET', 'POST'])
def configura_modelo():
    # Carrega os dados temporários de novo
    arq_temp_caminho = session.get('arquivo_temporario')
    if arq_temp_caminho is None:
        return redirect(url_for('index'))

    data = pd.read_csv(arq_temp_caminho)

    # Recupera as colunas selecionadas
    X = data[[session['bairro_col'], session['metragem_col'], session['quartos_col'],
              session['banheiros_col'], session['tipo_col'],
              session['cidade_col'], session['tipo_col'], session['latitude_col'],
              session['longitude_col']]]
    y = data[session['preco_col']]

    if request.method == 'POST':
        # Configura o modelo com base nas escolhas do usuário
        modelo_escolhido = request.form.get("modelo")
        if modelo_escolhido == 'RandomForest':
            n_estimators = int(request.form.get("n_estimators", 100))
            max_depth = int(request.form.get("max_depth", 5))
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif modelo_escolhido == 'KNN':
            n_neighbors = int(request.form.get("n_neighbors", 3))
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            return "Modelo não suportado"

        # Divide os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Treina o modelo
        model.fit(X_train, y_train)

        # Faz predições e calcula a acurácia
        y_pred = model.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)

        return f"Modelo treinado com acurácia: {acuracia}"

    return render_template('configura_modelo.html')


if __name__ == '__main__':
    app.run(debug=True)
