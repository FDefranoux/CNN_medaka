# USAGE

# import the necessary packages
from pyimagesearch import config
if config.MLFLOW:
    import mlflow
from pyimagesearch.classifier import Larval_MLPhenotyper
import sklearn
from pyimagesearch.ziramutils import get_dataloader, train_val_split, ZiramDataset 
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


def main():

####################### PARAMS settings (to be put in config.py) #####################################

    # Recuperation and printing and saving the parameters of the run
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    config.modif_params_argparse()
    hyperparam = config.HYPERPARAMS
    
    yaml_file=open(os.path.join(config.OUTPUT_PATH, "run_info.yaml"),"w")
    yaml.dump(hyperparam, yaml_file)
    yaml_file.close()

    run_name = hyperparam.pop('NAME')
    exp_name = hyperparam.pop('EXP_NAME')
    print(f'[INFO] Parameters of the run ({run_name}):\n')
    print(pd.DataFrame(hyperparam, index=['Value']).T.to_markdown(), '\n')

    if config.MLFLOW:
        print(f'\n\n[INFO] Starting MLFLOW run ({run_name}) on {exp_name} experiment\n')
        mlflow.set_tracking_uri(config.ML_OUTPUT)
        mlflow.set_experiment(exp_name)
        mlflow.start_run(run_name=run_name)
        mlflow.log_artifact(os.path.join(ABS_PATH, 'pyimagesearch', 'config.py'))
        mlflow.log_artifact(os.path.join(config.OUTPUT_PATH, "run_info.yaml"))
        for key, item in hyperparam.items():
            mlflow.log_param(key, item)
    
    if config.PLOT_AUROC:
        df_train = pd.DataFrame(columns=['class_' + str(n) for n in range(hyperparam['NUM_CLASS'])], index=range(1, hyperparam['EPOCHS']+1, 1))
        df_random = pd.DataFrame(columns=['class_' + str(n) for n in range(hyperparam['NUM_CLASS'])], index=range(1, hyperparam['EPOCHS']+1, 1))
        df_val = pd.DataFrame(columns=['class_' + str(n) for n in range(hyperparam['NUM_CLASS'])], index=range(1, hyperparam['EPOCHS']+1, 1))

#################### Initialisation of training ################################

    # Creating and Loading the datasets
    trainDataset = ZiramDataset(path=hyperparam['BASE_PATH'],
                                dataset='training_set', 
                                filename = hyperparam['FILENAME'],
                                label=hyperparam['LABEL'],
                                num_class=hyperparam['NUM_CLASS'],
                                im_size=hyperparam['IMAGE_SIZE'],
                                mean=config.MEAN, 
                                std=config.STD,
                                augment=True)

    (trainDataset, valDataset) = train_val_split(dataset=trainDataset, valSplit=0.2)  # training and validation data split
    trainLoader = get_dataloader(trainDataset, hyperparam['BATCH_SIZE'])  # training data loader per batch
    valLoader = get_dataloader(valDataset, hyperparam['BATCH_SIZE'])  # validation data loader per batch

    # build the custom model
    model = Larval_MLPhenotyper(numClasses=hyperparam['NUM_CLASS']).to(config.DEVICE)

    # initialize loss function (criterion) and optimizer
    criterion_classif = torch.nn.CrossEntropyLoss()
    criterion_regression = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparam['LR'])
    best_valauroc, n_max= 0, 0
    dict_auroc = {'Training':[], 'Validation':[], 'Random':[]}

############################## TRAINING #########################################
    
    # loop over epochs
    for epoch in range(hyperparam['EPOCHS']):
        print(f'\nEpoch {epoch + 1 } / {hyperparam["EPOCHS"]}')
        
        model.train()  # set the model in training mode

        train_metrics = MetricRecorder(num_class=hyperparam['NUM_CLASS'],  prefix='Train', metric_ls=[])
        random_metrics = MetricRecorder(num_class=hyperparam['NUM_CLASS'],  prefix='Random', metric_ls=['Auroc'])

        # loop over the training set
        tqdm_object = tqdm(trainLoader, total=len(trainLoader), desc=f'Training', colour='#00ff00')
        for batch in tqdm_object:

            image_batch, target_class, target_reg = (batch['image'].to(config.DEVICE), batch['label_class'].to(config.DEVICE), batch['label_reg'].to(config.DEVICE))
            target_reg = torch.unsqueeze(target_reg, -1)

            # Compute loss
            out_dict = model(image_batch)
            logits_class, coefficient_reg = out_dict['logits'], out_dict['coefficient']
            random_logits = torch.from_numpy(np.random.rand(*logits_class.shape))

            # print(len(logits_class), len(target_class))
            classif_loss = criterion_classif(logits_class, target_class).type(torch.FloatTensor)
            regression_loss = criterion_regression(coefficient_reg, target_reg.type(torch.FloatTensor)) 

            # Update weight of model (retropagation)
            crit_loss = classif_loss + regression_loss
            optimizer.zero_grad()  # mise a zero des gradients
            crit_loss.type(torch.FloatTensor).backward()         # Calcul de la loss
            optimizer.step()        # calcul des gradient et retropagation

            random_metrics.metric_update(logits=random_logits, targets=target_class)
            train_metrics.metric_update(logits=logits_class, targets=target_class, loss_class=classif_loss, 
                                 loss_reg=regression_loss, loss_tot=crit_loss)
        

        mean_train_metrics = train_metrics.compute_array()
        mean_random_metrics = random_metrics.compute_array()
        print(mean_train_metrics)
        dict_auroc['Training'].append(mean_train_metrics)
        dict_auroc['Random'].append(mean_random_metrics)
############################### VALIDATION ################################################### 

        with torch.no_grad():  # Prevent the optimizer to compute gradients
            model.eval()  # set the model in evaluation mode
        
            val_metrics = MetricRecorder(num_class=hyperparam['NUM_CLASS'],  prefix='Validation', metric_ls=[])

            tqdm_object = tqdm(valLoader, total=len(valLoader), desc=f'Validation', colour='blue')
            for batch in tqdm_object:  # loop over the validation set

                # send the input to the device
                image_batch, target_class, target_reg = (batch['image'].to(config.DEVICE), batch['label_class'].to(config.DEVICE), batch['label_reg'].to(config.DEVICE))
                target_reg = torch.unsqueeze(target_reg, -1)
                
                # make the predictions and calculate the validation loss
                out_dict = model(image_batch)
                logits_class, coefficient_reg = out_dict['logits'], out_dict['coefficient']

                classif_loss = criterion_classif(logits_class, target_class)
                regression_loss = criterion_regression(coefficient_reg, target_reg)
                crit_loss = classif_loss + regression_loss
                
                # pass the output logits through the softmax layer to get
                # output predictions, and calculate the number of correct
                # predictions
                val_metrics.metric_update(logits=logits_class, targets=target_class, loss_class=classif_loss, 
                                            loss_reg=regression_loss, loss_tot=crit_loss)
            
        mean_val_metrics = val_metrics.compute_array()
        dict_auroc['Validation'].append(mean_val_metrics)
        print(mean_val_metrics)
            

####################### RESULTS saving ###########################################

        # Compute and update results

        
        # MLflow save training history
        if config.MLFLOW:
            train_metrics.save_mlflow(epoch=epoch+1, mean=False)
            val_metrics.save_mlflow(epoch=epoch+1, mean=False)
            random_metrics.save_mlflow(epoch=epoch+1, mean=True)
        
        if config.PLOT_AUROC:
            # print(len(train_metrics), df_train.columns)
            df_train.loc[epoch+1] = train_metrics.dict_array['Auroc']
            df_val.loc[epoch+1] = val_metrics.dict_array['Auroc']
            df_random.loc[epoch+1] = random_metrics.dict_array['Auroc']

            # dict_auroc['train'].append(mean_train_metrics)
            # dict_auroc['random'].append(mean_random_metrics)
            # dict_auroc['val'].append(mean_val_metrics)
          
            

        # recording best weights of the run
        if mean_val_metrics['Auroc'] > best_valauroc:
            model_state_dict = model.state_dict()
            torch.save(model_state_dict, config.MODEL_PATH)
            best_state_dict = {
                # "model": model.state_dict(),
                # "optimizer": optimizer.state_dict(),
                "epoch": epoch +1,
                "auroc": mean_val_metrics['Auroc'] ,
            }
            best_valauroc = mean_val_metrics['Auroc']
        if mean_val_metrics['Auroc'].mean() > 0.99:
            n_max += 1
        else:
            n_max = 0
        if n_max >= 5: 
            print(f'[INFO] AUROC is at max, stopping the run after  {epoch +1 } epochs')
            break


######################### ENDING #####################################

    # serialize the model state to disk
    print('\n[INFO] Best Validation AUROC at epoch ', best_state_dict['epoch'] , ': ', best_state_dict['auroc'])
    # plot_results(H)
    
    if config.MLFLOW:
        try:
            mlflow.pytorch.log_state_dict(model_state_dict, artifact_path='best_auroc')
        except Exception as err:
            print( '\nERR state_dict : ', err)
        mlflow.end_run()
    
    if config.PLOT_AUROC:
        df_auroc = pd.DataFrame(dict_auroc).melt().reset_index()
        df['index'] = df['index'].astype(str)
        df_auroc['index'] = df_auroc['index'] + 1
        fig = sns.relplot(data=df_auroc,  x='index', y='value', hue='variable')
        try:
            fig.savefig(config.OUTPUT_PATH + 'AUROC_plot.jpg', bbox_inches='tight', dpi=100)
        except AttributeError:
            fig.get_figure().savefig(config.OUTPUT_PATH + 'AUROC_plot.jpg', bbox_inches='tight', dpi=100)
        print(f"\n({config.OUTPUT_PATH + 'AUROC_plot.jpg'})\n")



if __name__ == '__main__':
    main()
