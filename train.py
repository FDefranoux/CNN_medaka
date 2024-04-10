# USAGE

# import the necessary packages
from pyimagesearch import config
if config.MLFLOW:
    import mlflow
from pyimagesearch.classifier import Larval_MLPhenotyper
import sklearn
from pyimagesearch.ziramutils import get_dataloader, train_val_split, ZiramDataset, MetricRecorder
from torch import optim
from tqdm import tqdm 
import torch
import sys
import pandas as pd
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

def plot_results(H):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    # plot the training loss and accuracy
    df = pd.DataFrame(H)
    print(df.stack())
    df.to_csv(os.path.join(config.OUTPUT_PATH, 'run_values.csv'))
    df = df.stack().reset_index()
    df.columns = ['epoch', 'var', 'val']
    df['set'] = df['var'].str.replace('Loss', '').str.replace('Acc', '')
    df['type'] = df['var'].str.replace('train', '').str.replace('val', '')
    df['epoch'] = df['epoch'] + 1
    g = sns.relplot(kind='line', data=df, x='epoch', y='val', row='type',facet_kws={'sharey':False} )
    g.savefig('model_training_seaborn.jpg', bbox_inches='tight', dpi=100)

    plt.figure()
    plt.plot(H["trainLoss"], label="train_loss")
    plt.plot(H["valLoss"], label="val_loss")
    plt.plot(H["trainAcc"], label="train_acc")
    plt.plot(H["valAcc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(config.OUTPUT_PATH, "model_training.png"))


class AvgMeter:
    """Small helper recording a moving average."""
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def main():

####################### PARAMS settings #####################################

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
    best_valauroc = 0
    n_max = 0

############################## TRAINING #########################################
    
    # loop over epochs
    for epoch in range(hyperparam['EPOCHS']):
        print(f'\nEpoch {epoch + 1 } / {hyperparam["EPOCHS"]}')
        
        model.train()  # set the model in training mode

        # initialize the total training and validation loss
        classTrainLoss, classValLoss, regTrainLoss, regValLoss, totalTrainLoss, totalValLoss = 0, 0, 0, 0, 0, 0
        trainCorrect, valCorrect = 0, 0
        trainAuroc = torchmetrics.AUROC(task="multiclass", num_classes=hyperparam['NUM_CLASS'], average=None)
        valAuroc = torchmetrics.AUROC(task="multiclass", num_classes=hyperparam['NUM_CLASS'], average=None)
        randomAuroc = torchmetrics.AUROC(task="multiclass", num_classes=hyperparam['NUM_CLASS'], average=None)
        reg_loss = AvgMeter()
        avg_regloss = torchmetrics.aggregation.MeanMetric()
        # f1score = torchmetrics.---(task="multiclass", num_classes=hyperparam['NUM_CLASS'], average=None)

        # loop over the training set
        # for (image, target) in tqdm(trainLoader, desc=f'\nEpoch {epoch + 1}/{hyperparam["EPOCHS"]}'):
        tqdm_object = tqdm(trainLoader, total=len(trainLoader), desc=f'Training', colour='#00ff00')
        for batch in tqdm_object:

            image_batch, target_class, target_reg = (batch['image'].to(config.DEVICE), batch['label_class'].to(config.DEVICE), batch['label_reg'].to(config.DEVICE))
            target_reg = torch.unsqueeze(target_reg, -1)

            # Compute loss
            out_dict = model(image_batch)
            logits, coefficient = out_dict['logits'], out_dict['coefficient']
            random_logits = torch.from_numpy(np.random.rand(*logits.shape))
            classif_loss = criterion_classif(logits, target_class).type(torch.FloatTensor)
            regression_loss = criterion_regression(coefficient, target_reg.type(torch.FloatTensor)) 

            # Update weight of model (retropagation)
            crit_loss = classif_loss + regression_loss
            optimizer.zero_grad()  # mise a zero des gradients
            crit_loss.type(torch.FloatTensor).backward()         # Calcul de la loss
            optimizer.step()        # calcul des gradient et retropagation

            #####  Compute metrics
            totalTrainLoss += crit_loss.item()
            classTrainLoss += classif_loss.item()
            regTrainLoss += regression_loss.item()
            print(regression_loss.item(), regTrainLoss)

            reg_loss.update(regTrainLoss)
            # loss_avg = loss.compute().cpu().numpy()
            avg_regloss.update(regTrainLoss)
            avg_regloss_value = avg_regloss.compute().cpu().numpy()

            trainCorrect += (logits.argmax(dim=-1) == target_class).sum().item()
            trainAuroc.update(torch.nn.functional.softmax(logits, dim=-1), target_class)
            trainAuroc_value = trainAuroc.compute().cpu().numpy()

            randomAuroc.update(torch.nn.functional.softmax(random_logits, dim=-1), target_class)
            randomAuroc_value = randomAuroc.compute().cpu().numpy()

            
            tqdm_object.set_postfix(
                                    TrainAuroc=trainAuroc_value.mean(),
                                    Random_auroc=randomAuroc_value.mean(),
                                    reg_loss = reg_loss.avg, 
                                    totalTrainloss = totalTrainLoss,
                                    # loss_avg = loss_avg.mean(), 
                                    avg_regloss = avg_regloss_value.mean()

                                )

############################### VALIDATION ################################################### 

        with torch.no_grad():  # Prevent the optimizer to compute gradients
            model.eval()  # set the model in evaluation mode
        
            reg_loss = AvgMeter()

            tqdm_object = tqdm(valLoader, total=len(valLoader), desc=f'Validation', colour='blue')
            for batch in tqdm_object:  # loop over the validation set

                # send the input to the device
                image_batch, target_class, target_reg = (batch['image'].to(config.DEVICE), batch['label_class'].to(config.DEVICE), batch['label_reg'].to(config.DEVICE))
                target_reg = torch.unsqueeze(target_reg, -1)
                # make the predictions and calculate the validation loss
                out_dict = model(image_batch)
                logits, coefficient = out_dict['logits'], out_dict['coefficient']

                classif_loss = criterion_classif(logits, target_class)
                regression_loss = criterion_regression(coefficient, target_reg)
                valLoss = classif_loss + regression_loss
                
                
                totalValLoss += valLoss.item()
                classValLoss += classif_loss.item()
                regValLoss += regression_loss.item()

                # pass the output logits through the softmax layer to get
                # output predictions, and calculate the number of correct
                # predictions
                valCorrect += (logits.argmax(dim=-1) == target_class).sum().item()
                valAuroc.update(torch.nn.functional.softmax(logits, dim=-1), target_class)
                valAuroc_value = valAuroc.compute().cpu().numpy()
                reg_loss.update(regValLoss)

                tqdm_object.set_postfix(
                                        ValAuroc=valAuroc_value.mean(),
                                        Random_auroc=randomAuroc_value.mean(),
                                        reg_loss = reg_loss.avg, 
                                    )
        

####################### RESULTS saving ###########################################

        # Compute and update results
    
        # calculate the average training and validation loss
        train_size = (len(trainDataset) // hyperparam['BATCH_SIZE'] + 1)
        val_size = (len(valDataset) // hyperparam['BATCH_SIZE'] + 1)

        avgTrainLoss = totalTrainLoss / train_size
        avgclassTrainLoss = classTrainLoss / train_size
        avgregTrainLoss = regTrainLoss / train_size

        avgValLoss = totalValLoss / val_size
        avgclassValLoss = classValLoss / val_size
        avgregValLoss = regValLoss / val_size

        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataset)
        valCorrect = valCorrect / len(valDataset)

        
        # MLflow save training history
        if config.MLFLOW:
            mlflow.log_metric(key='trainLoss', value=avgTrainLoss, step=epoch)
            mlflow.log_metric(key='avgclassTrainLoss', value=avgclassTrainLoss, step=epoch)
            mlflow.log_metric(key='avgregTrainLoss', value=avgregTrainLoss, step=epoch)

            mlflow.log_metric(key='valLoss', value=avgValLoss, step=epoch)
            mlflow.log_metric(key='avgclassValLoss', value=avgclassValLoss, step=epoch)
            mlflow.log_metric(key='avgregValLoss', value=avgregValLoss, step=epoch)
            mlflow.log_metric(key='trainAcc', value=trainCorrect, step=epoch)
            mlflow.log_metric(key='valAcc', value=valCorrect, step=epoch)
            
            if hyperparam['NUM_CLASS'] > 2:
                print(f'[INFO] AUROC per class: training = {trainAuroc_value}, validation = {valAuroc_value}')
                
            for n in range(hyperparam['NUM_CLASS']): 
                mlflow.log_metric(key=f'randomAuroc_{n}', value=float(randomAuroc_value[n]), step=epoch)
                mlflow.log_metric(key=f'trainAuroc_{n}', value=float(trainAuroc_value[n]), step=epoch)
                mlflow.log_metric(key=f'valAuroc_{n}', value=float(valAuroc_value[n]), step=epoch)

        # recording best weights of the run
        if valAuroc_value.mean() > best_valauroc:
            model_state_dict = model.state_dict()
            torch.save(model_state_dict, config.MODEL_PATH)
            best_state_dict = {
                # "model": model.state_dict(),
                # "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "auroc": valAuroc_value.mean(),
            }
            best_valauroc = valAuroc_value.mean()
        if valAuroc_value.mean() > 0.99:
            n_max += 1
        else:
            n_max = 0
        if n_max >= 5: 
            print(f'[INFO] AUROC is at max, stopping the run after  {epoch +1 } epochs')
            break

        # # print the model training and validation information
        # print(f"\n[SUMMARY] EPOCH: {epoch + 1}/{hyperparam['EPOCHS']}")
        # print(pd.DataFrame([trainAuroc_value, valAuroc_value, randomAuroc_value],
        #         columns=['Auroc'], index=['train', 'Validation', 'random']).T.to_markdown(), '\n')
        # print(f"Auroc: {avgTrainLoss:.6f}, Train accuracy: {trainCorrect:.4f}")
        # print(f"Val loss: {avgValLoss:.6f}, Val accuracy: {valCorrect:.4f}")

######################### ENDING #####################################

    # serialize the model state to disk
    print('\n[INFO] Best Validation AUROC at epoch ', best_state_dict['epoch'] + 1, ': ', best_state_dict['auroc'])
    # plot_results(H)
    
    if config.MLFLOW:
        try:
            mlflow.pytorch.log_state_dict(model_state_dict, artifact_path='best_auroc')
        except Exception as err:
            print( '\nERR state_dict : ', err)
        mlflow.end_run()


if __name__ == '__main__':
    main()