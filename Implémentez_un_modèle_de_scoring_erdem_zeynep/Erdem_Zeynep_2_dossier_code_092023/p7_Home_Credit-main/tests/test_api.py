

import unittest
from fastapi import status
import requests
import json
class TestTools(unittest.TestCase):
  

    def setup(self):
        self.url = "https://apihomecredit-861d00eaed91.herokuapp.com/" 
        #self.url = "http://localhost:8000/"
        self.client_data_json = "[{\"EXT_SOURCE_2\":0.5202729458,\"EXT_SOURCE_3\":null,\"CODE_GENDER\":1,\"PAYMENT_RATE\":0.0482779168,\"DAYS_EMPLOYED\":null,\"INSTAL_DPD_MEAN\":0.5333333333,\"PREV_CNT_PAYMENT_MEAN\":12.0,\"NAME_EDUCATION_TYPE_Highereducation\":1,\"DAYS_BIRTH\":55.8,\"AMT_ANNUITY\":36459.0}]"

    def test_homepage(self):
        url = "https://apihomecredit-861d00eaed91.herokuapp.com/" 
        #url = "http://localhost:8000/"
        response = requests.get(url)
        assert response.status_code == 200
        assert json.loads(response.content) == 'Welcome to API Pret a depenser'

    def test_prediction(self):
       
        url = "https://apihomecredit-861d00eaed91.herokuapp.com/predict" 
        #url = "http://localhost:8000/predict"
        client_data_json = "[{\"EXT_SOURCE_2\":0.5202729458,\"EXT_SOURCE_3\":null,\"CODE_GENDER\":1,\"PAYMENT_RATE\":0.0482779168,\"DAYS_EMPLOYED\":null,\"INSTAL_DPD_MEAN\":0.5333333333,\"PREV_CNT_PAYMENT_MEAN\":12.0,\"NAME_EDUCATION_TYPE_Highereducation\":1,\"DAYS_BIRTH\":55.8,\"AMT_ANNUITY\":36459.0}]"
        response = requests.post(url, json={"data": client_data_json})
        prediction = response.json()["proba"]
        expected_result = 0.37
        assert prediction == expected_result
        
    def test_shap_local(self):
        url = "https://apihomecredit-861d00eaed91.herokuapp.com/shaplocal" 
        #url = "http://localhost:8000/shaplocal"
        client_data_json = "[{\"EXT_SOURCE_2\":0.5202729458,\"EXT_SOURCE_3\":null,\"CODE_GENDER\":1,\"PAYMENT_RATE\":0.0482779168,\"DAYS_EMPLOYED\":null,\"INSTAL_DPD_MEAN\":0.5333333333,\"PREV_CNT_PAYMENT_MEAN\":12.0,\"NAME_EDUCATION_TYPE_Highereducation\":1,\"DAYS_BIRTH\":55.8,\"AMT_ANNUITY\":36459.0}]"
        response = requests.post(url, json={"data": client_data_json})
        res = json.loads(response.content)
        base_values = res['base_value']
        expected_result = -0.4876034355259544
        assert base_values == expected_result
        