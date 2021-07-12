import argparse
import logging
import pandas as pd
from polyaxon_client.tracking import Experiment, get_log_level, get_outputs_path

from src.modelling.model import Model
from src.datapipeline.loader import Datapipeline #import datapipeline class

logger = logging.getLogger(__name__)


def run_experiment(train_data_path, test_data_path,embedding_data_path):
    try:
        log_level = get_log_level()
        if not log_level:
            log_level = logging.INFO

        logger.info("Starting experiment")

        experiment = Experiment()
        logging.basicConfig(level=log_level)

        dpl = Datapipeline()

        # Load your data from your datapipeline
        X_train, y_train, X_val,y_val, word_index_train = dpl.transform_train_data(train_data_path) # transform your train data using the train_data_path as a parameter

        X_test, y_test = dpl.transform_test_data(test_data_path) # transform your test data using the test_data_path as a parameter

        model = Model(embedding_data_path,word_index_train)
        
        train_acc = model.train(X_train, y_train,X_val,y_val)

        logger.info("Training acc logged")
        experiment.log_metrics(acc=train_acc)


        #saving model to ouput tab

        model.save(experiment.get_outputs_path() + "/model.h5")
        
        #just do training first??

        #test_acc = model.evaluate(X_test, y_test)

        #logger.info("output acc logged")
        #experiment.log_metrics(acc=test_acc)

        #metrics = model.train(params)

        #experiment.log_metrics(**metrics)

        

        logger.info("Experiment completed")
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")


if __name__ == '__main__':
    #in_params = {}  # Add your own params
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path',action="store",default="<fill in default path>")
    parser.add_argument('--test_data_path',action="store",default="<fill in default path>")
    parser.add_argument('--embedding_data_path',action="store",default="<fill in default path>")
    parser.add_argument('--model_data_path',action="store",default="<fill in default path>")


    # Also ensure you set the default values to the best model you have
    args = parser.parse_args()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    embedding_data_path = args.embedding_data_path
    model_data_path = args.model_data_path
    


    #params = {} 
    run_experiment(train_data_path, test_data_path,embedding_data_path)



