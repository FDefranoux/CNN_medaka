import torch
import torchmetrics
import mlflow
from mlflow_extend import mlflow as mlflow_extend
import pandas as pd


class MetricRecorder:
    """Small helper recording targeted metrics."""
    # TODO Own plotting ? 
    # TODO Fonction to save best dict 
    
    def __init__(self, num_class, prefix='', metric_ls=[]):
        self.num_class = num_class
        self.prefix = prefix
        dict_metrics = {
            'Auroc' : torchmetrics.AUROC(task="multiclass", num_classes=num_class, average=None),
            'Accuracy' : torchmetrics.Accuracy(task="multiclass", num_classes=num_class, average=None),
            'Loss' : torchmetrics.aggregation.MeanMetric(),
            'Conf_matrix': torchmetrics.ConfusionMatrix(num_classes=num_class),
            'R2score' : torchmetrics.regression.R2Score()
        }
        # if self.dict_metrics: self.metric_reset()
        self.dict_metrics = {key: metric for (key, metric) in dict_metrics.items() if key in metric_ls}
        if  len(self.dict_metrics.keys()) < 1 :
            raise AttributeError("No metrics specified or corresponding")

    def metric_reset(self):
        for name, metric in self.dict_metrics.items():
            self.dict_metrics[name] = metric.reset()

    def metric_update(self, logits, targets, loss=None):
        # Here loss should not necessary, you can compile directly the loss with logits and targets (see how criterion is calculated in model)
        preds = torch.nn.functional.softmax(logits, dim=-1)
        if 'Auroc' in self.dict_metrics.keys(): self.dict_metrics['Auroc'].update(preds, targets)
        if 'Conf_matrix' in self.dict_metrics.keys(): self.dict_metrics['Conf_matrix'].update(preds, targets)
        if 'R2score' in self.dict_metrics.keys(): self.dict_metrics['R2score'].update(logits, targets)
        #TODO: FOR accuracy do you need the preds or the logits ???
        if 'Accuracy' in self.dict_metrics.keys(): self.dict_metrics['Accuracy'].update(logits, targets)
        if 'Loss' in self.dict_metrics.keys(): self.dict_metrics['Loss'].update(loss)

    def compute_array(self, mean=False, array=True):
        self.dict_array = {}
        if mean: self.dict_mean = {}
        for name, metric in self.dict_metrics.items():
            if array: 
                self.dict_array[name] = metric.compute().cpu().numpy()
            if (mean==True) & (array==False) : 
                self.dict_mean[name] =  metric.compute().cpu().numpy().mean()
            if (mean==True) & (array==True):    
                self.dict_mean[name] = self.dict_array[name].mean()
        
    def save_mlflow(self, epoch, mean=False):
        mlflow.log_metric(key=f'step', value=epoch, step=epoch)
        for name, array in self.dict_array.items():
            if name == 'Conf_matrix':
                try:
                    mlflow_extend.log_confusion_matrix(array)
                except: 
                    print(f'Could not save the confusion matrix\n {epoch}, \n{array}')
            else:
                if array.size == 1:
                    mlflow.log_metric(key=f'{self.prefix}_{name}', value=float(array), step=epoch)
                else:
                    for n in range(self.num_class):
                        mlflow.log_metric(key=f'{self.prefix}_{name}_{n}', value=float(array[n]), step=epoch)
        if mean:
            for name, array in self.dict_mean.items():
                mlflow.log_metric(key=f'{self.prefix}_{name}_mean', value=float(array.mean()), step=epoch)

        

