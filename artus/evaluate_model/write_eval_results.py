"""Export plots or csv after COCO-style evaluation of a trained deep learning model.

A csv file can be exported or merged to an already existed one. This csv contains 
AP, AP50, AP75, APm, APl and AP-perclass. Plots can also be exported after having exporting this csv.
The plots created are multiple histograms where each bar stands for a model.
"""

import pandas as pd
import plotly.express as px
import os

class ModelsMetricsFormat():
    """ A class that writes the evaluation results in a CSV format.
    
    If the CSV already exists, it will merge the results at the last row creating a dataframe where
    each row is one model evaluated along with their metrics.
    
    Attributes:
        session (str): the name of the directory containing a model_final.pth to evaluate
        eval_results (dict): the evaluation results from detectron2.trainer.test()
        export_dir (str): the path to the directory where metrics will be exported or
            where previous metrics.csv are saved.
        csv_name (str): a csv name to save the results of evaluation
    
    """
    def __init__(self, session, eval_results, export_dir, csv_name):
        """ Initializes the instance.

        Export_directory and csv_name are join in a complete csv_path.
        
        Args:
            session (str): the name of the directory containing a model_final.pth to evaluate
            eval_results (dict): the evaluation results from detectron2.trainer.test()
            export_dir (str): the path to the directory where metrics will be exported or
                where previous metrics.csv are saved.
            csv_name (str): a csv name to save the results of evaluation
        """
        self.session = session
        self.accuracy = eval_results
        self.csv_path = os.path.join(export_dir, csv_name)

    def check_csv_exists(self):
        """Check if a csv containing model's metrics exists or not.

        If csv already exists,results of this session will be appended to this csv. 
        Otherwise, a csv will be created.
        """
        file_exists = os.path.exists(self.csv_path)
        if file_exists:
            print('CSV containing metrics already exists. Results will be appended to this dataframe')
        else:
            print('CSV does not exist yet. We will create it for you.')
        return file_exists

    def to_pandas(self):
        """Take results from COCOevaluator and format it into a pandas dataframe."""
        segmentation_accuracies = list(self.accuracy.values())[1]
        accuracy_df = pd.DataFrame(data=segmentation_accuracies.values(), index=list(segmentation_accuracies.keys()))
        accuracy_df = accuracy_df.transpose()
        accuracy_df['model'] = self.session

        if self.check_csv_exists():
            models_metrics = pd.read_csv(self.csv_path, header=0)
            accuracy_df = pd.concat([models_metrics, accuracy_df], axis=0)

        return accuracy_df
    
    def write_to_csv(self):
        """Export results to csv format at the csv_path"""
        df = self.to_pandas()
        df.to_csv(self.csv_path, index=False)

class ModelsMetricsPlots():
    """ A class that plots the results of different models trained.

    The plots created are multiple histograms where each bar stands for a model.
    Plots are more relevant if the csv contains multiple models metrics.

    Attributes: 
        csv_metrics_path (str): the path to a csv dataframe.
        export_dir (str): the directory where the interactive plots will be exported
        plot_name (str): the name of the plot file with .html extension
        title (str): title to display on plot
    """
    
    def __init__(self, csv_metrics_path, export_dir, plot_name, title):
        """Initializes the instance.

        Args:        
            csv_metrics_path (str): the path to a csv dataframe. The results of Pÿthon(ModelsMetricsFormat.write_to_csv())
            The dataframe should have the following columns : model | AP | AP50 | AP75 | APs | APm | APl | AP-class1 | AP-class2 etc...
            One row represents the evaluation metrics of one model.
            export_dir (str): the directory where the interactive plots will be exported
            plot_name (str): the name of the plot file  with the extension (html file)
            title (str): title to display on plot
        """
        self.csv_metrics_path = csv_metrics_path
        self.export_dir = export_dir
        self.plot_name = plot_name
        self.title = title

    def process_csv(self):
        """ Read csv and pivot the dataframe.

        Create a dataframe with one row for each variable and each model.
        """
        metrics = pd.read_csv(self.csv_metrics_path, header=0)
        
        variables = metrics.columns.drop(['model'])
        
        metrics_melted = pd.melt(
            metrics, 
            id_vars='model', 
            value_vars=variables,
            value_name='Average precision')
        
        metrics_melted.dropna(axis=0, how='any', inplace=True)
        return metrics_melted

    def plot_metrics(self):
        """Plot the metrics for all the models described in the csv file.

        Plots created are facet bar plots. One facet is created per variables (AP50, Ap-class1 etc...)
        and each model precision is described by one bar.

        Returns:
            A plotly interactive barplot that can be seen with Python(fig.show())
        """
        metrics = self.process_csv()

        fig = px.bar(metrics, x="model", y="Average precision", color="model", barmode="relative",
             facet_col="variable", facet_col_wrap=5, facet_row_spacing=0.04, facet_col_spacing=0.04,
             height=1500, width = 1500, title= self.title)
        
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        return fig

    def export_plots(self):
        """Export the plots in html format. Plots are interactives."""
        fig = self.plot_metrics()
        fig.write_html(os.path.join(self.export_dir, self.plot_name))


    
        

