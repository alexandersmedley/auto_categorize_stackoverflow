from flask import Flask, request, jsonify, make_response
from flask_restx import Api, Resource, fields
import joblib

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from gensim import utils
import gensim.parsing.preprocessing as gsp

# Define and apply gensim filters
filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
#            gsp.strip_short, 
           gsp.stem_text
          ]

def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s

# Custom transformer using gensim filters
class TextCleaner(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
    

flask_app = Flask(__name__)

app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "StackOverflow tag predictor", 
		  description = "Suggests tags given a StackOverflow posts")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params', 
                  {'title': fields.String(required = True, 
                                           description="StackOverflow post title", 
                                           help="Text Field 1 cannot be blank"), 
                   'body': fields.String(required = True, 
                                            description="StackOverflow post body", 
                                            help="Text Field 1 cannot be blank")
                  }
                 )

with open('supervised_model_maxdf.pkl', 'rb') as f:
    classifier = joblib.load(f)

with open('supervised_model_maxdf_tags.csv', 'rb') as f:
    model_tags = pd.read_csv(f, sep = ';')

@name_space.route("/")
class MainClass(Resource):
    
    def options(self):
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response
    
    @app.expect(model)		
    def post(self):
        try: 
            formData = request.json
#            data = [val for val in formData.values()]
            data = [formData['title'] + ' ' + formData['body']]
            
            y_proba = classifier.predict_proba(data)
            y_proba = pd.DataFrame(y_proba.transpose().toarray())
            y_tags = model_tags.loc[y_proba[y_proba[0] > 0.2].index, :]['tag'].to_list()
            y_tags_str = ', '.join(y_tags)
            
            response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": "Prediction: " + y_tags_str 
				})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as error:
            return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})

if __name__ == '__main__':
    flask_app.run(host = '0.0.0.0', port = 5000, debug = False)
#    flask_app.run()