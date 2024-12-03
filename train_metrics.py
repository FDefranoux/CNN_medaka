# USAGE

# import the necessary packages
from pyimagesearch import config
from pyimagesearch.classifier import Larval_MLClassifier
import sklearn
from pyimagesearch.ziramutils import ZiramDataset, train_val_split
from pyimagesearch.metrics import MetricRecorder
from torch import optim
from tqdm import tqdm 
import torch
import sys
import pandas as pd
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
from pyimagesearch.config import params_fromYAML
import argparse

def main(yaml_file):

    train_kwargs, model_kwargs, hyperparams, params, metrics_kwargs = params_fromYAML(yaml_file)
    numClasses = train_kwargs.pop('num_class')
    numEpochs = params['EPOCHS']

####################### PARAMS settings (to be put in config.py) #####################################
    
    # # Recuperation and printing and saving the parameters of the run
    # ABS_PATH = o, flush=Trues.path.dirname(os.path.abspath(__file__))
    # config.modif_params_argparse()
    # # hyperparam = config.HYPERPARAMS
    # yaml_file=open(os.path.join(config.OUTPUT_PATH, "run_info.yaml"),"w")
    # yaml.dump(hyperparam, yaml_file)
    # yaml_file.close()

    if params['MLFLOW']:
        import mlflow
        print(f'\n\n[INFO] Starting MLFLOW run\n', flush=True)
        mlflow.set_tracking_uri(params['MLFLOW_OUT'])
        mlflow.set_experiment(params['EXP_NAME'])
        mlflow.start_run(run_name=params['NAME'])

        mlflow.log_artifact(os.path.join(params['OUTPUT_PATH'], "run_info.yaml"))
        for key, item in hyperparams.items():
            print('MLFLOW', key)
            mlflow.log_param(key, item)
        supp_params = dict(model=model_kwargs['model_dict']['model'],
                            filename=train_kwargs['filename'],
                            label=train_kwargs['label'],
                            model_mode=model_kwargs['mode'], 
                            set='training')
        
        for key, item in supp_params.items():
            print('MLFLOW', key)
            mlflow.log_param(key, item)
    


#################### Initialisation of training ################################

    # Creating and Loading the datasets
    split_kwargs = {key: train_kwargs.pop(key) for key in ['validation_ratio', 'batch_size', 'shuffle']}
    Dataset = ZiramDataset(mode='training', **train_kwargs)
    assert int(Dataset.num_label) == int(numClasses), f'Numbers of labels ({Dataset.num_label}) not corresponding to class number ({numClasses})'

    # if params['PLOT_AUROC']:
    #     df_train = pd.DataFrame(columns=['class_' + str(n) for n in range(numClasses)], index=range(1, numEpochs +1, 1))
    #     df_random = pd.DataFrame(columns=['class_' + str(n) for n in range(numClasses)], index=range(1, numEpochs +1, 1))
    #     df_val = pd.DataFrame(columns=['class_' + str(n) for n in range(numClasses)], index=range(1, numEpochs +1, 1))    
    

    trainLoader, valLoader = train_val_split(dataset=Dataset, **split_kwargs)

    
    ### IF PROBLEM UNCOOMENT THIS BLOCK
    # trainDataset = ZiramDataset(path=hyperparam['BASE_PATH'],
    #                             dataset='training_set', 
    #                             filename = hyperparam['FILENAME'],
    #                             label=hyperparam['LABEL'],
    #                             num_class=numClasses,
    #                             im_size=hyperparam['IMAGE_SIZE'],
    #                             mean=config.MEAN, 
    #                             std=config.STD,
    #                             augment=True)
    # (trainDataset, valDataset) = train_val_split(dataset=trainDataset, valSplit=params.VALIDATION_RATIO)  # training and validation data split
    # trainLoader = get_dataloader(trainDataset, hyperparams['BATCH_SIZE'])  # training data loader per batch
    # valLoader = get_dataloader(valDataset, hyperparams['BATCH_SIZE'])  # validation data loader per batch

    # build the custom model
    model = Larval_MLClassifier(**model_kwargs).to(params['DEVICE'])
    # model = Larval_MLClassifier(numClasses=hyperparams['NUM_CLASS']).to(params['DEVICE'])

    # initialize loss function (criterion) and optimizer
    criterion = model.criterion()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['LR'])
    if model_kwargs['mode'] == 'categorical':
        best_metric, n_max= 0, 0
    elif model_kwargs['mode'] == 'regression':
        best_metric, n_max= 100, 0
    if params['PLOT_AUROC']: dict_auroc = {'Training':[], 'Validation':[], 'Random':[]}

############################## TRAINING #########################################
    
    # loop over epochs
    for epoch in range(numEpochs):
        print(f'\nEpoch {epoch + 1 } / {numEpochs}', flush=True)
        
        model.train()  # set the model in training mode

        train_metrics = MetricRecorder(num_class=numClasses,  prefix='Train', metric_ls=metrics_kwargs['TRAINING'])
        random_metrics = MetricRecorder(num_class=numClasses,  prefix='Random', metric_ls=metrics_kwargs['RANDOM'])

        # loop over the training set
        tqdm_object = tqdm(trainLoader, total=len(trainLoader), desc=f'Training', colour='#00ff00')
        for batch in tqdm_object:

            image_batch, target = (batch['image'].to(params['DEVICE']), batch['label_' + str(model_kwargs["mode"])].to(params['DEVICE']))
            if  model_kwargs['mode'] == 'regression':
                target = torch.unsqueeze(target, -1).to(params['DEVICE'])
                print(target)

            

            # Compute loss
            out_dict = model(image_batch)
            logits = out_dict['logits'].to(params['DEVICE'])
            random_logits = torch.from_numpy(np.random.rand(*logits.shape)).to(params['DEVICE'])
            if  model_kwargs['mode'] == 'regression':
                crit_loss = criterion(logits, target.type(torch.FloatTensor)).to(params['DEVICE'])
            else:
                crit_loss = criterion(logits, target).to(params['DEVICE'])

            # Update weight of model (retropagation)
            optimizer.zero_grad()  # mise a zero des gradients
            if  model_kwargs['mode'] == 'regression':
                crit_loss.type(torch.FloatTensor).backward()         # Calcul de la loss
                target = target.type(torch.IntTensor)
            else:
                crit_loss.backward()  

            optimizer.step()        # calcul des gradient et retropagation

            if model_kwargs['mode'] == 'categorical': random_metrics.metric_update(logits=random_logits, targets=target)
            train_metrics.metric_update(logits=logits, targets=target, loss=crit_loss)
        
        train_metrics.compute_array(mean=True)
        if params['PLOT_AUROC']: dict_auroc['Training'].append(train_metrics.dict_mean['Auroc'])
        if model_kwargs['mode'] == 'categorical': 
            random_metrics.compute_array(mean=True, array=False)
            if params['PLOT_AUROC']: dict_auroc['Random'].append(random_metrics.dict_mean['Auroc'])

############################### VALIDATION ################################################### 

        with torch.no_grad():  # Prevent the optimizer to compute gradients
            model.eval()  # set the model in evaluation mode
        
            val_metrics = MetricRecorder(num_class=numClasses,  prefix='Validation', metric_ls=metrics_kwargs['VALIDATION'])

            tqdm_object = tqdm(valLoader, total=len(valLoader), desc=f'Validation', colour='blue')
            for batch in tqdm_object:  # loop over the validation set

                # send the input to the device
                image_batch, target = (batch['image'].to(params['DEVICE']), batch['label_' + str(model_kwargs["mode"])].to(params['DEVICE']))
                if model_kwargs['mode'] == 'regression':
                    target = torch.unsqueeze(target, -1).to(params['DEVICE'])
                    target = target.type(torch.IntTensor)
                    
                
                # make the predictions and calculate the validation loss
                out_dict = model(image_batch)
                logits = out_dict['logits'].to(params['DEVICE'])
                crit_loss = criterion(logits, target)

                
                # pass the output logits through the softmax layer to get
                # output predictions, and calculate the number of correct
                # predictions
                val_metrics.metric_update(logits=logits, targets=target, loss=crit_loss)
            
        val_metrics.compute_array(mean=True)
        if params['PLOT_AUROC']: dict_auroc['Validation'].append(val_metrics.dict_mean['Auroc'])

####################### RESULTS saving ###########################################

        # Compute and update results
        # MLflow save training history
        if params['MLFLOW']:
            train_metrics.save_mlflow(epoch=epoch+1, mean=False)
            val_metrics.save_mlflow(epoch=epoch+1, mean=False)
            if model_kwargs['mode'] == 'categorical': random_metrics.save_mlflow(epoch=epoch+1, mean=True)

        # recording best weights of the run
        if model_kwargs['mode'] == 'categorical': 
            if val_metrics.dict_mean['Auroc'] > best_metric:
                model_state_dict = model.state_dict()
                torch.save(model_state_dict, os.path.join(params['OUTPUT_PATH'], 'model.pth'))
                best_state_dict = {
                    # "model": model.state_dict(),
                    # "optimizer": optimizer.state_dict(),
                    "epoch": epoch +1,
                    "metric": val_metrics.dict_mean['Auroc'] ,
                }
                best_metric = val_metrics.dict_mean['Auroc']
            if val_metrics.dict_mean['Auroc'].mean() > 0.99:
                n_max += 1
            else:
                n_max = 0
            if n_max >= 5: 
                print(f'[INFO] AUROC is at max, stopping the run after  {epoch +1 } epochs', flush=True)
                break

        elif model_kwargs['mode'] == 'regression':
            if val_metrics.dict_mean['Loss'] < best_metric:
                model_state_dict = model.state_dict()
                torch.save(model_state_dict, os.path.join(params['OUTPUT_PATH'], 'model.pth'))
                best_state_dict = {
                    # "model": model.state_dict(),
                    # "optimizer": optimizer.state_dict(),
                    "epoch": epoch +1,
                    "metric": val_metrics.dict_mean['Loss'] ,
                }
                best_metric = val_metrics.dict_mean['Loss']
            if round(val_metrics.dict_mean['Loss'].mean(), 3) == round(best_metric, 3):
                n_max += 1
            else:
                n_max = 0
            if n_max >= 5: 
                print(f'[INFO] Loss is at min, stopping the run after  {epoch +1 } epochs', flush=True)
                break            


######################### ENDING #####################################

    # serialize the model state to disk
    print('\n[INFO] Best Training at epoch ', best_state_dict['epoch'] , ': ', best_state_dict['metric'], flush=True)
    # plot_results(H)
    
    if params['MLFLOW']:
        try:
            mlflow.pytorch.log_state_dict(model_state_dict, artifact_path='best_metric')
            mlflow.log_param('status', 'finished')
        except Exception as err:
            print( '\nERR state_dict : ' + str(err), flush=True )
        mlflow.end_run()
    
    if params['PLOT_AUROC']:
        df_auroc = pd.DataFrame(dict_auroc)
        df_auroc['epochs'] = df_auroc.index + 1
        df_auroc['epochs'] = df_auroc['epochs'].astype(str)
        df_auroc = df_auroc.melt(id_vars=['epochs'])
        fig = sns.relplot(data=df_auroc,  x='epochs', y='value', hue='variable', kind='line', aspect=7)
        try:
            fig.savefig(os.path.join(params['OUTPUT_PATH'], 'AUROC_plot.jpg'), bbox_inches='tight', dpi=100)
        except AttributeError:
            fig.get_figure().savefig(os.path.join(params['OUTPUT_PATH'], 'AUROC_plot.jpg'), bbox_inches='tight', dpi=100)
        print(f"\n({os.path.join(params['OUTPUT_PATH'], 'AUROC_plot.jpg')})\n", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("-Y", "--yaml_file", type=str, help='Path of the YAML file containing the necessary information to run the training')
    args = parser.parse_args()
    main(yaml_file = args.yaml_file)
