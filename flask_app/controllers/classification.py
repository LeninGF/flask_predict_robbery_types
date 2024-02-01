from flask import render_template, request
from flask_app import app
from flask_app.models.classifier import RobberyAi
from flask_app.models.report import Table

@app.route('/')
def welcome():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = {"text":request.form['text_input']}
    processed_text = RobberyAi.preprocess_text(data['text'])
    y_hat = RobberyAi.predict(data['text'])
    y_hat_delitos_validados = RobberyAi.predict_delitos_validados(data['text'])
    response = {'input':data['text'],
                'prediction_del_seguimeinto':y_hat,
                'prediction_del_validados':y_hat_delitos_validados}
    # print(response)
    return render_template("prediction.html", 
                            # relato=data['text'],
                            relato=processed_text,
                            clase=y_hat[0]['label'],
                            probabilidad=y_hat[0]['score']*100,
                            clase_validados=y_hat_delitos_validados[0]['label'],
                            probabilidad_validados=y_hat_delitos_validados[0]['score']*100)


@app.route('/demo')
def show_demo():
    table_result = Table.table_generator()
    return render_template("Demo.html",
                            column_names = table_result.columns.values,
                            row_data = list(table_result.values.tolist()),
                            zip = zip)

@app.route('/indicet')
def home():
    return render_template("home2.html")
@app.route('/roboai')
def roboai():
    return render_template("comoFunciona.html")


@app.route('/tokenizeme', methods=['GET', 'POST'])
def show_tokenization():
    batch = {"relato":request.form['text_input']}
    print(batch)
    embeddings = RobberyAi.tokenize_through_pipe(batch["relato"])
    print(embeddings)
    return render_template("comoFunciona.html", embeddings=embeddings)

