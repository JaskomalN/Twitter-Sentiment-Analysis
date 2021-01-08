
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
import pickle


tweet_df = pd.read_csv("train_preprocessed.csv")

tweet_df["clean_text"]=tweet_df["clean_text"].astype('U')

pipeline = Pipeline([
   ( 'bow',CountVectorizer()),
    ('classifier',MultinomialNB()),
])

from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test = train_test_split(tweet_df['clean_text'],tweet_df['Sentiment'],test_size=0.3)

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

print(classification_report(predictions, label_test))


pickle.dump(model_pipeline,open("model_pipeline.pkl","wb"))
