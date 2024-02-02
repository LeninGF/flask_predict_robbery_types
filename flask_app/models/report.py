from flask_app.models.classifier import RobberyAi
import time
import pandas as pd


class Table:
    PATH_TEST_SET = '/home/falconiel/CodePrograms/clasificaion_robos_fge/data/processed/validacionJunio2022.csv' # delitos_seguimiento
    PATH_MODEL_DELITOS_VALIDADOS = '/home/falconiel/ML_Models/robbery_validados_tf20231211' #delitos_validados
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
        self.etiqueta_prediccion_delitos_validados = data['etiqueta_prediccion_delitos_validados']
        self.prob_score_delitos_validados = data['prob_score_delitos_validados']
    @classmethod
    def table_generator(cls):
        xtest = cls.xtest
        xtest_random = xtest.sample(10)
        print(xtest_random.columns)
        # +++++ delitos seguimiento +++++
        xtest_random["etiqueta_prediccion"], xtest_random["score"], xtest_random["tiempo(ms)"] = zip(*xtest_random.RELATO.map(cls.prediction))
        # print(xtest_random[['delitos_seguimiento', 'etiqueta_prediccion', 'score', 'tiempo(ms)']])
        xtest_random["Status"] = xtest_random.apply(lambda row: cls.check_status(row.delitos_seguimiento, row.etiqueta_prediccion), axis=1)
        
        # +++++ delitos validados +++++
        xtest_random["etiqueta_prediccion_delitos_validados"], xtest_random["score_delitos_validados"], xtest_random["tiempo(ms)_delitos_validados"] = zip(*xtest_random.RELATO.map(cls.prediction_delitos_validados))
        xtest_random["Status_wrt_delitos_validados"] = xtest_random.apply(lambda row: cls.check_status(row.delitos_validados, row.etiqueta_prediccion_delitos_validados), axis=1)
        
        table_result = xtest_random[['NDD', 
                                     'Tipo_Delito_PJ', 
                                     'RELATO', 
                                     'desagregacion', 
                                     'delitos_seguimiento', 
                                     'etiqueta_prediccion', 
                                     'score', 
                                     'tiempo(ms)', 
                                     'Status', 
                                     'delitos_validados', 
                                     'etiqueta_prediccion_delitos_validados', 
                                     'score_delitos_validados',
                                     'tiempo(ms)_delitos_validados',
                                     'Status_wrt_delitos_validados']]
                                     
        # formating the table
        table_result['tiempo(ms)'] = table_result['tiempo(ms)'].apply(lambda x : float("{:.2f}".format(x)))
        table_result['tiempo(ms)_delitos_validados'] = table_result['tiempo(ms)_delitos_validados'].apply(lambda x : float("{:.2f}".format(x)))
        table_result.score = table_result.score.apply(lambda x : "{:.2f}%".format(x))
        table_result.score_delitos_validados = table_result.score_delitos_validados.apply(lambda x : "{:.2f}%".format(x))
        table_result.rename(columns={'desagregacion':'desagregacion_fge'}, inplace=True)
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
    def prediction_delitos_validados(x):
        t_start = time.time()
        y_hat = RobberyAi.predict_delitos_validados(x)
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
