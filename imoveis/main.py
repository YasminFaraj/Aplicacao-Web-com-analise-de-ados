from flask import request, render_template, redirect, url_for, session, flash
from flask_login import login_user, logout_user, login_required
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pandas as pd
from config import app
from models import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import base64
import tempfile
import plotly.express as px
import joblib


# função para limpar as colunas com valores nulos
def limpar_colunas_nulas(data, preco_col):
    # remove colunas com valores nulos em X (colunas que não sejam o preço e que não foram selecionadas)
    X = data.drop(columns=[preco_col])
    X = X.dropna(axis=1, how='any')  # remove colunas com valores nulos

    # converte strings em colunas categóricas para valores numéricos
    for col in X.columns:
        if X[col].dtype == 'object':
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])

    # remove linhas onde y (preço) está nulo
    y = data[preco_col]
    X = X[y.notnull()]
    y = y.dropna()

    return X, y


# função para buscar parâmetros otimizados
def otimizar_parametros(modelo, X_train, y_train, parametros):
    grid_search = GridSearchCV(estimator=modelo, param_grid=parametros, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # mostrar os melhores parâmetros encontrados
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

    return grid_search.best_estimator_


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    from forms import RegisterForm
    from werkzeug.security import generate_password_hash
    from flask import flash

    formulario = RegisterForm()

    # verifica se os dados enviados são válidos
    if formulario.validate_on_submit(): # se foram válidos ele entra no if
        usuario = formulario.username.data 
        senha = generate_password_hash(formulario.password.data) 
        # o generate_password_hash() pega a senha informada pelo usuário e criptografa ela
        # print(f'Senha criptografada: {senha}') 

        usuario_existe = User.query.filter_by(username=usuario).first() # verifica se o usuário existe

        # if user_exists == True:
        if usuario_existe:
            flash('Erro: O usuário já existe!', 'error')
        else:
            novo_usuario = User(username=usuario, password=senha)

            db.session.add(novo_usuario)
            db.session.commit()

            flash('Sucesso: O usuário foi criado!', 'success')

            return redirect(url_for('login'))

    return render_template('register.html', form=formulario)


@app.route('/login', methods=['GET', 'POST'])
def login():
    from forms import LoginForm
    from werkzeug.security import check_password_hash
    from flask import flash

    formulario = LoginForm()

    if formulario.validate_on_submit():
        usuario = formulario.username.data
        usuario_db = User.query.filter_by(username=usuario).first()

        if usuario_db:
            senha = formulario.password.data
            senha_db = usuario_db.password

            # print(f'Senha informada: {senha}')  # debugging
            # print(f'Senha no banco: {senha_db}')  # debugging

            if check_password_hash(senha_db, senha):
                flash('Login bem-sucedido', 'success')
                login_user(usuario_db)
                return redirect(url_for('home'))
            else:
                flash('Senha incorreta', 'error')
        else:
            flash('Usuário não encontrado', 'error')

    return render_template('login.html', form=formulario)


# esse código desloga o usuário atual e o redireciona para a página de login
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))


# página inicial para upload do arquivo CSV
@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
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
    
    return render_template('home.html')


# página para seleção de colunas
@app.route('/selecao_colunas', methods=['GET', 'POST'])
def selecao_colunas():
    # carrega o dataframe da sessão (os dados temporários)
    arq_temp_caminho = session.get('arquivo_temporario')
    if arq_temp_caminho is None:
        return redirect(url_for('home'))

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

        # valida se há pelo menos 4 colunas, além do preço
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

        # cria um novo arquivo temporário com os dados filtrados
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as arq_filtrado:
            X.to_csv(arq_filtrado.name, index=False)
            session['arquivo_filtrado'] = arq_filtrado.name
        
        # Verifica qual botão foi clicado
        action = request.form.get('action')
        if action == 'configuracao':
            # Redireciona para a página de configuração do modelo
            return redirect(url_for('configura_modelo'))
        elif action == 'graficos':
            # Redireciona para a página de gráficos
            return redirect(url_for('graficos'))

        # Caso nenhuma ação válida seja identificada
        flash("Erro: Ação inválida.", "error")
        return redirect(url_for('selecao_colunas'))

    return render_template('selecao_colunas.html', columns=colunas)


# página para configuração do modelo e treinamento
@app.route('/configura_modelo', methods=['GET', 'POST'])
def configura_modelo():
    # carrega os dados temporários de novo
    arq_filtrado_caminho = session.get('arquivo_filtrado')
    if arq_filtrado_caminho is None:
        return redirect(url_for('home'))

    data = pd.read_csv(arq_filtrado_caminho)

    # chama a função para limpar as colunas nula
    X, y = limpar_colunas_nulas(data, session['preco_col'])

    # Armazena as colunas de X para uso posterior
    session['colunas_X'] = X.columns.tolist()

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

        # buscar os melhores parâmetros com GridSearchCV
        #best_model = otimizar_parametros(model, X_train, y_train, parametros)

        print("Iniciando o treinamento com o modelo:", modelo_escolhido)
        print("Dados de treinamento:", X_train.shape, y_train.shape)
        model.fit(X_train, y_train)
        print("Modelo treinado com sucesso!")

        # treina o modelo
        model.fit(X_train, y_train)
        # treina o modelo com os melhores parâmetros (grid)
        #best_model.fit(X_train, y_train)

        # faz predições e calcula a acurácia
        y_pred = model.predict(X_test)
        # faz predições e calcula a acurácia (grid)
        #y_pred = best_model.predict(X_test

        # Salva o modelo treinado em um arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as modelo_temp:
            joblib.dump(model, modelo_temp.name)  # Salva o modelo usando joblib
            session['modelo_treinado_path'] = modelo_temp.name  # Armazena o caminho no session

        # calcula RMSE e R²
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        flash(f"Treinamento concluído! RMSE: {rmse:.2f}, R²: {r2:.4f}", "success")
        return redirect(url_for('resultado_previsao'))

    return render_template('configura_modelo.html')


@app.route('/resultado_previsao', methods=['GET', 'POST'])
def resultado_previsao():
    # Carrega os dados filtrados e o modelo treinado
    modelo_treinado_path = session.get('modelo_treinado_path')
    if modelo_treinado_path is None:
        flash("Erro: Nenhum modelo treinado encontrado.", "error")
        return redirect(url_for('configura_modelo'))

    modelo_treinado = joblib.load(modelo_treinado_path)
    if modelo_treinado is None:
        flash("Erro: Falha ao carregar o modelo treinado.", "error")
        return redirect(url_for('configura_modelo'))

    # Recupera as colunas selecionadas
    colunas_selecionadas = [
        session.get('bairro_col'),
        session.get('metragem_col'),
        session.get('quartos_col'),
        session.get('banheiros_col'),
        session.get('tipo_col'),
        session.get('cidade_col')
    ]
    colunas_selecionadas = [col for col in colunas_selecionadas if col]

    # Verifique se colunas_selecionadas não está vazia
    if not colunas_selecionadas:
        flash("Erro: Nenhuma coluna válida foi selecionada.", "error")
        return redirect(url_for('selecao_colunas'))

    # Se for POST, faz a previsão com base nos inputs fornecidos
    if request.method == 'POST':
        try:
            dados_usuario_teste = pd.DataFrame([{
                'quartos_col': 3,  # Exemplo: 3 quartos
                'banheiros_col': 2,  # Exemplo: 2 banheiros
                'metragem_col': 80,  # Exemplo: 80m²
                'bairro_col': 'Manhattan',  # Exemplo de valor categórico
                'tipo_col': 'House for sale',
                'cidade_col': 'New York',
                'latitude_col': '406743632',
                'longitude_col': '-739587248'
            }])

            # Codificação das variáveis categóricas
            dados_usuario_teste = pd.get_dummies(dados_usuario_teste, drop_first=True)

            # Garantir que as colunas estão em conformidade com as do modelo
            colunas_modelo = session.get('colunas_X')
            for col in colunas_modelo:
                if col not in dados_usuario_teste.columns:
                    dados_usuario_teste[col] = 0  # Preencher com zero ou NA

            dados_usuario_teste = dados_usuario_teste.reindex(columns=colunas_modelo, fill_value=0)

            previsao_teste = modelo_treinado.predict(dados_usuario_teste)
            print("Previsão para dados de teste:", previsao_teste)  # Verifique se o modelo gera previsões



            # Lê os valores fornecidos pelo usuário para as colunas selecionadas
            inputs_usuario = {col: request.form.get(col) for col in colunas_selecionadas}
            print(inputs_usuario)  # Imprima os valores para verificar

            if any(value is None or value == "" for value in inputs_usuario.values()):
                flash("Erro: Todos os campos devem ser preenchidos.", "error")
                return render_template('resultado_previsao.html', colunas=colunas_selecionadas)

            # Verifica se os campos de quartos e banheiros têm valores válidos
            try:
                inputs_usuario['quartos_col'] = float(inputs_usuario.get('quartos_col', 0))
                inputs_usuario['metragem_col'] = float(inputs_usuario.get('metragem_col', 0))
                inputs_usuario['banheiros_col'] = float(inputs_usuario.get('banheiros_col', 0))
            except ValueError:
                flash("Erro: Quartos e Banheiros devem ser números válidos.", "error")
                return render_template('resultado_previsao.html', colunas=colunas_selecionadas)

            # Converte para DataFrame e trata tipos de dados
            dados_usuario = pd.DataFrame([inputs_usuario])

            # Codifica as variáveis categóricas usando One-Hot Encoding
            dados_usuario = pd.get_dummies(dados_usuario, drop_first=True)

            colunas_modelo = session.get('colunas_X')
            if colunas_modelo is None:
                flash("Erro: As colunas do modelo não estão disponíveis.", "error")
                return redirect(url_for('selecao_colunas'))

            for col in colunas_modelo:
                if col not in dados_usuario.columns:
                    dados_usuario[col] = 0  # Preencher com zero ou NA

            dados_usuario = dados_usuario.reindex(columns=colunas_modelo, fill_value=0)

            previsao = modelo_treinado.predict(dados_usuario)
            flash(f"O valor previsto para o imóvel é: R$ {previsao[0]:,.2f}", "success")

        except Exception as e:
            flash(f"Erro ao realizar a previsão: {str(e)}", "error")
            return render_template('resultado_previsao.html', colunas=colunas_selecionadas)

    # Exibe a página com as colunas necessárias para a previsão
    return render_template('resultado_previsao.html', colunas=colunas_selecionadas)


@app.route('/graficos', methods=['GET'])
def graficos():
    # carrega os dados temporários de novo
    arq_filtrado_caminho = session.get('arquivo_filtrado')
    if arq_filtrado_caminho is None:
        return redirect(url_for('index'))

    data = pd.read_csv(arq_filtrado_caminho)

    # verifica se as colunas de latitude, logitude, preço, metragem foram selecionadas
    latitude_col = session.get('latitude_col')
    longitude_col = session.get('longitude_col')
    if not latitude_col or not longitude_col or latitude_col not in data.columns or longitude_col not in data.columns:
        flash("Erro: As colunas de latitude e longitude são necessárias para gerar o mapa.", "error")
        return redirect(url_for('selecao_colunas'))
    
    preco_col = session.get('preco_col')
    if not preco_col or preco_col not in data.columns:
        flash("Erro: A coluna 'preço' não foi selecionada para gerar gráficos.", "error")
        return redirect(url_for('selecao_colunas'))
    
    metragem_col = session.get('metragem_col')
    if not metragem_col or metragem_col not in data.columns:
        flash("Erro: A coluna 'metragem' não foi selecionada para gerar gráficos.", "error")
        return redirect(url_for('selecao_colunas'))

    # grafico de pizza - distribuicao por faixa de preco
    faixas = ['Até 100 mil', '100 mil - 300 mil', '300 mil - 500 mil', '500 mil - 1 milhão', 'Mais de 1 milhão']
    bins = [0, 100000, 300000, 500000, 1000000, float('inf')]
    data['Faixa de Preço'] = pd.cut(data[preco_col], bins=bins, labels=faixas, include_lowest=True)
    distribucao = data['Faixa de Preço'].value_counts()

    plt.figure(figsize=(8, 6))
    plt.pie(distribucao, labels=distribucao.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribuição por Faixa de Preço')

    # converte o gráfico para base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    imagem_base64_pizza = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    # gráfico interativo de barra (plotly) - distribuicao por faixa de preco
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=distribucao.index,
        y=distribucao.values,
        marker_color='skyblue'
    ))
    fig_bar.update_layout(
        title='Distribuição por Faixa de Preço',
        xaxis_title='Faixa de Preço',
        yaxis_title='Quantidade',
        template='plotly_dark'
    )
    fig_bar_html = fig_bar.to_html(full_html=False)

    # configura o token (substitua pela sua chave do Mapbox)
    mapbox_token = "SEU_MAPBOX_TOKEN_AQUI"
    px.set_mapbox_access_token(mapbox_token)

    # adicionar informacoes para os marcadores
    data['tooltip'] = data.apply(
        lambda row: f"Tipo: {row.get(session.get('tipo_col'), 'N/A')}<br>"
                    f"Cidade: {row.get(session.get('cidade_col'), 'N/A')}<br>"
                    f"Bairro: {row.get(session.get('bairro_col'), 'N/A')}<br>"
                    f"Preço: R${row.get(session.get('preco_col'), 'N/A'):.2f}",
        axis=1
    )

    # cria o grafico de mapa
    fig_map = px.scatter_mapbox(
        data,
        lat=latitude_col,
        lon=longitude_col,
        color=session.get('preco_col'),  # cor com base no preço
        size=session.get('metragem_col'),  # tamanho com base na metragem
        text='tooltip',  # informações para o hover
        hover_name='tooltip',  # detalhes para quando passar o mouse por cima
        title="Mapa Interativo de Imóveis",
        color_continuous_scale="Viridis",
        size_max=15,
        zoom=10,  # ajuste o nível de zoom (inicial)
        range_color=[0, 3_000_000]
    )

    fig_map.update_layout(mapbox_style="open-street-map")
    
    fig_map_html = fig_map.to_html(full_html=False)

    return render_template('graficos.html', 
                           imagem_base64_pizza=imagem_base64_pizza, 
                           fig_bar_html=fig_bar_html,
                           fig_map_html=fig_map_html)

if __name__ == '__main__':
    app.run(debug=True)
