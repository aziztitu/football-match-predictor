from django.shortcuts import render
from django.http import HttpResponse, HttpRequest, JsonResponse
from os import path
from joblib import dump, load
from time import time
import pandas as pd
import numpy as np
import json

# Create your views here.
def index(request: HttpRequest):

    result = {
        'success': False,
        'data': {},
        'msg': '',
    }

    classifierModels = {
        'lr': 'lr_classifier.model',
        'nb': 'nb_classifier.model',
        'rf': 'rf_classifier.model',
    }

    try:
        selectedClassifier = request.GET['classifier']

        if selectedClassifier not in classifierModels:
            result['success'] = False
            result['msg'] = 'Invalid Classifier'
            return HttpResponse(json.dumps(result))

        exportedModelFilePath = path.abspath(path.dirname(
            __name__)) + '/../exportedModels/' + classifierModels[selectedClassifier]
        # print(exportMetaDataFilePath)

        X = pd.DataFrame({
            'home_encoded': request.GET['homeTeam'],
            'away_encoded': request.GET['awayTeam'],
            'HTHG': request.GET['homeGoals'],
            'HTAG': request.GET['awayGoals'],
            'HS': request.GET['homeShots'],
            'AS': request.GET['awayShots'],
            'HST': request.GET['homeShotsOnTarget'],
            'AST': request.GET['awayShotsOnTarget'],
            'HR': request.GET['homeRedCards'],
            'AR': request.GET['awayRedCards'],
        }, index=[0])

        if path.exists(exportedModelFilePath):
            clf = load(exportedModelFilePath)
            # print(exportMetaData)

            start = time()
            pred_probs = clf.predict_proba(X)[0]

            y = dict(zip(clf.classes_, pred_probs))
            end = time()
            print("Made Predictions in {:2f} seconds".format(end-start))

            print("Result: ")
            print(y)

            result['data']['homeWin'] = y['H']
            result['data']['draw'] = y['D']
            result['data']['awayWin'] = y['A']

            result['success'] = True
            result['msg'] = 'Match Result Predicted'
        else:
            result['success'] = False
            result['msg'] = 'Classifier not found'

    except:
        result['success'] = False
        result['msg'] = 'Please check your input'

    return JsonResponse(result)
