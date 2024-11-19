from flask import Flask, request, render_template, redirect, url_for, session, flash
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import pandas as pd
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # limita a 16MB o tamamho do arquivo
app.secret_key = "your_secret_key"  # necessário para usar sessões


# Função para limpar as colunas com valores nulos
def limpar_colunas_nulas(data, preco_col):
    # Remove colunas com valores nulos em X (colunas que não sejam o preço e que não foram selecionadas)
    X = data.drop(columns=[preco_col])
    X = X.dropna(axis=1, how='any')  # remove colunas com valores nulos

    # Converte strings em colunas categóricas para valores numéricos
    for col in X.columns:
        if X[col].dtype == 'object':
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])

    # Remove linhas onde y (preço) está nulo
    y = data[preco_col]
    X = X[y.notnull()]
    y = y.dropna()

    return X, y


# Função para buscar parâmetros otimizados
def otimizar_parametros(modelo, X_train, y_train, parametros):
    grid_search = GridSearchCV(estimator=modelo, param_grid=parametros, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Mostrar os melhores parâmetros encontrados
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

    return grid_search.best_estimator_


# página inicial para upload do arquivo CSV
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session.clear()  # limpa qualquer dado armazenado na sessão
        if 'arquivo' not in request.files:
            return "Nenhum arquivo foi enviado"

        arquivo = request.files['arquivo']

        if arquivo.filename == '':
            return "Escolha um arquivo válido"

        # lê o arquivo CSV
        data = pd.read_csv(arquivo)

        # o arquivo estava grande, então essa parte cria um arquivo temporário para armazenas o csv
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as arq_temp:
            data.to_csv(arq_temp.name, index=False)
            session['arquivo_temporario'] = arq_temp.name  # armazena o caminho do arquivo temporário na sessão

        # redireciona para a página de seleção de colunas, passando as colunas do CSV
        return redirect(url_for('selecao_colunas'))

    return render_template('index.html')


# página para seleção de colunas
@app.route('/selecao_colunas', methods=['GET', 'POST'])
def selecao_colunas():
    # carrega o dataframe da sessão (os dados temporários)
    arq_temp_caminho = session.get('arquivo_temporario')
    if arq_temp_caminho is None:
        return redirect(url_for('index'))

    # lendo o arquivo e reconhecendo cada coluna
    data = pd.read_csv(arq_temp_caminho)
    colunas = data.columns

    if request.method == 'POST':
        # salva as colunas escolhidas pelo usuário ou None se "Nenhuma" for selecionada
        session['tipo_col'] = request.form['tipo'] if request.form['tipo'] else None
        session['cidade_col'] = request.form['cidade'] if request.form['cidade'] else None
        session['bairro_col'] = request.form['bairro'] if request.form['bairro'] else None
        session['metragem_col'] = request.form['metragem'] if request.form['metragem'] else None
        session['quartos_col'] = request.form['quartos'] if request.form['quartos'] else None
        session['banheiros_col'] = request.form['banheiros'] if request.form['banheiros'] else None
        session['preco_col'] = request.form['preco'] if request.form['preco'] else None
        session['latitude_col'] = request.form['latitude'] if request.form['latitude'] else None
        session['longitude_col'] = request.form['longitude'] if request.form['longitude'] else None

        # recupera as colunas selecionadas e filtra apenas as válidas
        colunas_selecionadas = {
            "bairro": session.get('bairro_col'),
            "metragem": session.get('metragem_col'),
            "quartos": session.get('quartos_col'),
            "banheiros": session.get('banheiros_col'),
            "tipo": session.get('tipo_col'),
            "cidade": session.get('cidade_col'),
            "latitude": session.get('latitude_col'),
            "longitude": session.get('longitude_col'),
            "preco": session.get('preco_col')
        }
        
        # filtra apenas colunas válidas (que não são None)
        colunas_validas = [coluna for coluna in colunas_selecionadas.values() if coluna is not None]

        # Valida se há pelo menos 4 colunas, além do preço
        if 0 < len(colunas_validas) <= 4 and session.get('preco_col') in colunas_validas:
            flash("Erro: Selecione pelo menos 5 colunas além da coluna de preço.", "error")
            return redirect(url_for('selecao_colunas'))

        # garante que pelo menos uma coluna foi selecionada
        if not colunas_validas:
            flash("Erro: Nenhuma coluna válida foi selecionada.", "error")
            return redirect('/selecao_colunas')

        if session.get('preco_col') not in colunas_validas:
            flash("Erro: A coluna de preço é obrigatória para treinar o modelo.", "error")
            return redirect('/selecao_colunas')

        # separa X e y do dataset
        X = data[colunas_validas]

        # Cria um novo arquivo temporário com os dados filtrados
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as arq_filtrado:
            X.to_csv(arq_filtrado.name, index=False)
            session['arquivo_filtrado'] = arq_filtrado.name

        # redireciona para a página de configuração do modelo
        return redirect(url_for('configura_modelo'))

    return render_template('selecao_colunas.html', columns=colunas)


# página para configuração do modelo e treinamento
@app.route('/configura_modelo', methods=['GET', 'POST'])
def configura_modelo():
    # carrega os dados temporários de novo
    arq_filtrado_caminho = session.get('arquivo_filtrado')
    if arq_filtrado_caminho is None:
        return redirect(url_for('index'))

    data = pd.read_csv(arq_filtrado_caminho)

    # Chama a função para limpar as colunas nula
    X, y = limpar_colunas_nulas(data, session['preco_col'])

    if request.method == 'POST':
        # configura o modelo com base nas escolhas do usuário
        modelo_escolhido = request.form.get("modelo")
        if modelo_escolhido == 'RandomForest':
            n_estimators = int(request.form.get("n_estimators", 115))  # Pega o valor de n_estimators
            max_depth = int(request.form.get("max_depth", 6))  # Pega o valor de max_depth
            min_samples_split = int(request.form.get("min_samples_split", 2))
            min_samples_leaf = int(request.form.get("min_samples_leaf", 1))
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            """parametros = {
                'n_estimators': [100, 115, 150, 200],
                'max_depth': [None, 5, 6, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }"""
        elif modelo_escolhido == 'KNN':
            n_neighbors = int(request.form.get("n_neighbors", 5))
            weights = request.form.get("weights", "uniform")
            model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights
            )
            """parametros = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }"""
        elif modelo_escolhido == 'DecisionTree':
            max_depth = int(request.form.get("max_depth_tree", 27))
            min_samples_split = int(request.form.get("min_samples_split_tree", 4))
            min_samples_leaf = int(request.form.get("min_samples_leaf_tree", 1))
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            """parametros = {
                'max_depth': [None, 10, 20, 27],
                'min_samples_split': [2, 4, 5],
                'min_samples_leaf': [1, 2, 4]
            }"""
        elif modelo_escolhido == 'XGBoost':  # Adicionando suporte ao XGBoost
            n_estimador = int(request.form.get("n_estimador", 115))
            learning_rate = float(request.form.get("learning_rate", 0.1))
            depth = int(request.form.get("depth", 5))
            model = XGBRegressor(
                n_estimators=n_estimador,
                learning_rate=learning_rate,
                max_depth=depth,
                random_state=42
            )
            """parametros = {
                'n_estimators': [50, 100, 115, 150, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 6, 10],
            }"""

        else:
            return "Modelo não suportado"

        # divide os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Buscar os melhores parâmetros com GridSearchCV
        #best_model = otimizar_parametros(model, X_train, y_train, parametros)

        # treina o modelo
        model.fit(X_train, y_train)
        # Treina o modelo com os melhores parâmetros (grid)
        #best_model.fit(X_train, y_train)

        # faz predições e calcula a acurácia
        y_pred = model.predict(X_test)
        # Faz predições e calcula a acurácia (grid)
        #y_pred = best_model.predict(X_test)

        # Calcula RMSE e R²
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return f"Modelo treinado com RMSE: {rmse:.4f} e R²: {r2:.4f}"

    return render_template('configura_modelo.html')


if __name__ == '__main__':
    app.run(debug=True)
