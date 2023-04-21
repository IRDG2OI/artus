import pandas as pd
import os

class ModelsMetricsFormat():
    ''' A function that writes the evaluation results in a CSV 'if the CSV already exists, it will merge the results at the last row).
    # Inputs:
    - session : the name of the directory containing a model_final.pth to evaluate
    - eval_results : the evaluation results from detectron2.trainer.test()
    - csv_name : a csv name to save the results of evaluation
    # Output:
    - a csv with the evaluation results or a row appended to a csv if the csv already exists
    '''
    def __init__(self, session, eval_results, export_dir, csv_name):
        self.session = session
        self.accuracy = eval_results
        self.csv_path = os.path.join(export_dir, csv_name)

    def check_csv_exists(self):
        '''Check if a csv containing model's metrics exists or not.
        If one exists, then results of this session will be appended to this csv. Otherwise, a csv will be created.'''
        file_exists = os.path.exists(self.csv_path)
        if file_exists:
            print('CSV containing metrics already exists. Results will be appended to this dataframe')
        else:
            print('CSV does not exist yet. We will create it for you.')
        return file_exists

    def to_pandas(self):
        '''Takes results from COCOevaluator and format it into a pandas dataframe.'''
        segmentation_accuracies = list(self.accuracy.values())[1]
        accuracy_df = pd.DataFrame(data=segmentation_accuracies.values(), index=list(segmentation_accuracies.keys()))
        accuracy_df = accuracy_df.transpose()
        accuracy_df['model'] = self.session

        if self.check_csv_exists():
            models_metrics = pd.read_csv(self.csv_path, header=0)
            accuracy_df = pd.concat([models_metrics, accuracy_df], axis=0)

        return accuracy_df
    
    def write_to_csv(self):
        df = self.to_pandas()
        df.to_csv(self.csv_path, index=False)
        

