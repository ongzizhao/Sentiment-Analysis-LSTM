import argparse
import logging
import pandas as pd
#from polyaxon_client.tracking import Experiment, get_log_level, get_outputs_path

from src.modelling.model import Model, predict_text
from src.datapipeline.loader import Datapipeline #import datapipeline class

logger = logging.getLogger(__name__)


def run_experiment(train_data_path, test_data_path,embedding_data_path):
    
    dpl = Datapipeline()

    # Load your data from your datapipeline
    X_train, y_train, X_val,y_val, word_index_train = dpl.transform_train_data(train_data_path) # transform your train data using the train_data_path as a parameter

    X_test, y_test = dpl.transform_test_data(test_data_path) # transform your test data using the test_data_path as a parameter

    model = Model(embedding_data_path,word_index_train)
    
    train_acc = model.train(X_train, y_train,X_val,y_val)

    #saving model to ouput tab
    
    #just do training first??

    #test_acc = model.evaluate(X_test, y_test)

    #logger.info("output acc logged")
    #experiment.log_metrics(acc=test_acc)

    #metrics = model.train(params)

    #experiment.log_metrics(**metrics)

    print(train_acc)

    text=["testing this is a test review, i am very unhappy with this place. The food is bland"]
    tokenizer_path = "./src/datapipeline/tokenizer.pickle"
    model_path = "./src/modelling/Model_checkpoint/model.h5"

    sentiment = predict_text(text, tokenizer_path, model_path)   

    print(sentiment)



if __name__ == '__main__':
    #in_params = {}  # Add your own params
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path',action="store",default="C:/Users/2nd pc/Desktop/MyProjects/AIAP/team1/data/train_50k.csv")
    parser.add_argument('--test_data_path',action="store",default="C:/Users/2nd pc/Desktop/MyProjects/AIAP/team1/data/test_50k.csv")
    parser.add_argument('--embedding_data_path',action="store",default="C:/Users/2nd pc/Desktop/MyProjects/AIAP/team1_test/embeddings_index_P.pickle")
    parser.add_argument('--model_data_path',action="store",default="C:/Users/2nd pc/Desktop/MyProjects/AIAP/team1_test/Model_checkpoint/Bi-LSTM.h5")


    # Also ensure you set the default values to the best model you have
    args = parser.parse_args()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    embedding_data_path = args.embedding_data_path
    model_data_path = args.model_data_path
    


    #params = {} 
    run_experiment(train_data_path, test_data_path,embedding_data_path)