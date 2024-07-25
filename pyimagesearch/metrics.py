import torch
import torchmetrics
import mlflow


class MetricRecorder:
    """Small helper recording a necessary metrics."""
    # TODO Own plotting ? 
    # TODO Fonction to save best dict 
    # TODO: Change to make it work with metric_kwargs
    
    def __init__(self, num_class, prefix='', metric_ls=[]):
        self.num_class = num_class
        self.prefix = prefix
        dict_metrics = {
            'Auroc' : torchmetrics.AUROC(task="multiclass", num_classes=num_class, average=None),
            'Accuracy' : torchmetrics.Accuracy(task="multiclass", num_classes=num_class, average=None),
            'Loss' : torchmetrics.aggregation.MeanMetric(),
        }
        # if self.dict_metrics: self.metric_reset()
        if metric_ls != []: self.dict_metrics = {key: metric for (key, metric) in dict_metrics.items() if key in metric_ls}
        else: self.dict_metrics = dict_metrics

    def metric_reset(self):
        for name, metric in self.dict_metrics.items():
            self.dict_metrics[name] = metric.reset()

    def metric_update(self, logits, targets, loss=None):
        if 'Auroc' in self.dict_metrics.keys(): self.dict_metrics['Auroc'].update(torch.nn.functional.softmax(logits, dim=-1), targets)
        if 'Accuracy' in self.dict_metrics.keys(): self.dict_metrics['Accuracy'].update(logits, targets)
        if 'Loss' in self.dict_metrics.keys(): self.dict_metrics['Loss'].update(loss)

    def compute_array(self):
        self.dict_array = {}
        self.dict_mean = {}
        for name, metric in self.dict_metrics.items():
            metric_array = metric.compute().cpu().numpy()
            self.dict_array[name] = metric_array
            self.dict_mean[name] = metric_array.mean()
        return self.dict_mean
        
    def save_mlflow(self, epoch, mean=False):
        for name, array in self.dict_array.items():
            if mean:
                mlflow.log_metric(key=f'{self.prefix}_{name}_mean', value=float(array.mean()), step=epoch)

            else:
                if array.size == 1:
                    mlflow.log_metric(key=f'{self.prefix}_{name}', value=float(array), step=epoch)
                else:
                    for n in range(self.num_class):
                        mlflow.log_metric(key=f'{self.prefix}_{name}_{n}', value=float(array[n]), step=epoch)
        

