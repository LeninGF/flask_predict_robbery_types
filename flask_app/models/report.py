from flask_app.models.classifier import RobberyAi
import time
import pandas as pd


class Table:
    PATH_TEST_SET = '/home/falconiel/CodePrograms/clasificaion_robos_fge/data/processed/validacionJunio2022.csv'
    xtest = pd.read_csv(PATH_TEST_SET, converters={'NDD':str})
    
    def __init__(self, data):
        self.id = data['id']
        self.ndd = data['Ndd']
        self.delito = data['delito']
        self.relato = data['relato']
        self.etiqueta_comision = data['etiqueta_comision']
        self.etiqueta_fge = data['etiqueta_fge']
        self.etiqueta_prediccion = data['etiqueta_prediccion']
        self.prob_score = data['prob_score']
    @classmethod
    def table_generator(cls):
        xtest = cls.xtest
        xtest_random = xtest.sample(10)
        # print(xtest_random.columns)
        xtest_random["etiqueta_prediccion"], xtest_random["score"], xtest_random["tiempo(ms)"] = zip(*xtest_random.RELATO.map(cls.prediction))
        # print(xtest_random[['delitos_seguimiento', 'etiqueta_prediccion', 'score', 'tiempo(ms)']])
        xtest_random["Status"] = xtest_random.apply(lambda row: cls.check_status(row.delitos_seguimiento, row.etiqueta_prediccion), axis=1)
        table_result = xtest_random[['NDD', 'Tipo_Delito_PJ', 'RELATO', 'desagregacion', 'delitos_seguimiento', 'etiqueta_prediccion', 'score', 'tiempo(ms)', 'Status']]
        # formating the table
        table_result['tiempo(ms)'] = table_result['tiempo(ms)'].apply(lambda x : float("{:.2f}".format(x)))
        table_result.score = table_result.score.apply(lambda x : "{:.2f}%".format(x))
        # print(table_result)
        return table_result
    
    @staticmethod
    def prediction(x):
        t_start = time.time()
        y_hat = RobberyAi.predict(x)
        t_delta = time.time() - t_start
        y_hat_dict = {'label': y_hat[0]['label'],
                    'score':y_hat[0]['score']*100,
                    'time':t_delta}
        label = y_hat_dict['label']
        score = y_hat_dict['score']
        time_ev = y_hat_dict['time']*1000
        return label, score, time_ev
        
    
    @staticmethod
    def check_status(y, yhat):
        if y == yhat:
            return "OK"
        else:
            return "Check"
