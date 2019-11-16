print('starting to load data')

from sklearn import model_selection, preprocessing, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
import sys, warnings
import pandas as pd

#set location to current location
__location__ = sys.path[0]
warnings.filterwarnings("ignore")

# load the dataset
trainDF = pd.read_excel('{}/test_task_DS.xlsx'.format(__location__), header=0)

print('data preprocessing')
#combine with product name and category
trainDF['text'] = trainDF[['description'
                             ,'initcat'
                             ,'case_size'
                             ,'pack_desc'
                            ]].apply(lambda x: ' '.join(str(x)),axis=1)

#labels
trainDF['label'] = trainDF.our_category

#fix labels misspelling
trainDF.label = trainDF.label.str.replace(' ans ',' and ')

# split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF, trainDF.label)

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

print('tf-idf matrix')
# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                                   ngram_range=(2,3),
                                   max_features=10000
                                  )
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x.text)
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x.text)

print('running classifier')
#define LogisticRegression classifier with predefined parameters
def lgr_pred(slvr, feature_vector_train, train_y, feature_vector_valid, valid_y):

    lgr = linear_model.LogisticRegression(solver=slvr
                                          , multi_class='multinomial').fit(feature_vector_train,train_y)

    # predict the labels on validation dataset
    prediction = lgr.predict(feature_vector_valid)

    # score the labels prediction on validation dataset
    score = lgr.score(feature_vector_valid,valid_y)
    print('classifier score on valid data is: ',score)

    # predict the probabilities of labels on validation dataset
    probabilities = lgr.predict_proba(feature_vector_valid)
    max_prob = [max(x) for x in probabilities]

    return score, max_prob, prediction

pred = lgr_pred('saga', xtrain_tfidf_ngram, train_y
                , xvalid_tfidf_ngram, valid_y)

#fit the train and validative df-s to the same shape
train_x['train_test'] = 'train'
train_x['y_predicted'] = ''
train_x['probability'] = -0.1
valid_x['train_test'] = 'test'
valid_x['y_predicted'] = encoder.inverse_transform(pred[2])
valid_x['probability'] = pred[1]

#union the train and validative df-s
final = pd.concat([train_x, valid_x])

print('loading results to excel')
#load data to excel
final[['case_size'
       ,'id'
       ,'pack_desc'
       ,'wholesale_price'
       ,'description'
       ,'train_test'
       ,'y_predicted'
       ,'probability']].to_excel("{}/test_task_DS_processed.xlsx".format(__location__)
                                 ,sheet_name='Sheet1'
                                 ,index=False)

print('complete')
