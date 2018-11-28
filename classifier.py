import numpy as np
import librosa
import sklearn
import os
import pickle 

if __name__ == "__main__":
    path_to_database = '/home/machon/voice_recognition/bazy/database'
    directory_list =  os.walk(path_to_database)
    features_list = []
    labels = []
    mfcc_list = []
    i = 0
    scaler = sklearn.preprocessing.StandardScaler()

    for directory in directory_list:
        if directory[0] != path_to_database:
            if True : #i == 91:
                for filename in os.listdir(directory[0]):
                    try:
                        y, sr = librosa.load(directory[0] + '/' + filename)
                        # print filename
                        mfcc = (librosa.feature.mfcc(y=y, sr=sr).T)
                        mfcc.mean(axis=0)
                        mfcc.std(axis=0)
                        mfcc_scaled = scaler.fit_transform(mfcc)
                        # print mfcc_scaled.shape
                        mfcc_list.append(mfcc_scaled)
                    except:
                        print 'Error reading ', filename, 'File corrupted'
                features = np.vstack((mfcc_list) )
                features_list.append(features)
                labels += [i]*len(features)
                # print len(features)
                # print features.shape
                # print len(labels)
                # features.append(mfcc)
                # labels.append(directory[0])
                # print labels
                # mfcc = []
                # if i == 2:
                #     break
            print i, filename
            i += 1
    print 'training started'
    
    svm = sklearn.svm.SVC(gamma='scale')
    svm.fit(np.vstack(features_list), labels)
    
    pickle.dump(svm, open('finalized_model.sav', 'wb'))