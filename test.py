import numpy as np
import librosa
import sklearn
import os
import pickle 


model = pickle.load(open('test_model.sav', 'rb'))
path_to_database = '/home/machon/voice_recognition/bazy/test'
directory_list =  os.walk(path_to_database)
features_list = []
labels = []
mfcc_list = []
res = []
i = 0
scaler = sklearn.preprocessing.StandardScaler()

for directory in directory_list:
        
    for filename in os.listdir(directory[0]):
        y, sr = librosa.load(directory[0] + '/' + filename)
        # print filename
        mfcc = (librosa.feature.mfcc(y=y, sr=sr).T)
        mfcc = mfcc.mean(axis=0)
        mfcc = mfcc.reshape(1,-1)
        # mfcc.std(axis=0)
        # mfcc_scaled = scaler.fit_transform(mfcc)
        # print mfcc_scaled.shape
        # mfcc_list.append(mfcc_scaled[0])
        # print mfcc
        print filename, model.predict(mfcc)