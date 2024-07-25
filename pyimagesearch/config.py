# import the necessary packages
import torch
import os
import argparse
import sys
import yaml
import pandas as pd

#QUESTION: Which params to be set up by argparse ?
#QUESTION: How to export the parameters to the training / evaluation scripts ?

# print(f'\n\n[INFO] Starting run ({run_name}) on {exp_name} experiment\n')
# print(f'[INFO] Parameters of the run ({run_name}):\n')
# print(pd.DataFrame(hyperparam, index=['Value']).T.to_markdown(), '\n')

# # determine the device type 
# if params.DEVICE == 'cuda':
#     if torch.cuda.is_available() == False: 
#         print('[WARNING] CUDA not available using CPU instead')
#         DEVICE = "cpu"
#     else:
#         DEVICE = torch.device("cuda")

# if config.MLFLOW:
#     mlflow.set_tracking_uri(config.ML_OUTPUT)
#     mlflow.set_experiment(exp_name)
#     mlflow.start_run(run_name=run_name)
#     mlflow.log_artifact(os.path.join(ABS_PATH, 'pyimagesearch', 'config.py'))
#     mlflow.log_artifact(os.path.join(config.OUTPUT_PATH, "run_info.yaml"))
#     for key, item in hyperparam.items():
#         mlflow.log_param(key, item)

# yaml_file=open(os.path.join(config.OUTPUT_PATH, "run_info.yaml"),"w")
# yaml.dump(hyperparam, yaml_file)
# yaml_file.close()



    # Recuperation and printing and saving the parameters of the run
    # config.modif_params_argparse()
    # hyperparam = config.HYPERPARAMS

    


    
    # if config.PLOT_AUROC:
    #     df_train = pd.DataFrame(columns=['class_' + str(n) for n in range(hyperparam['NUM_CLASS'])], index=range(1, hyperparam['EPOCHS']+1, 1))
    #     df_random = pd.DataFrame(columns=['class_' + str(n) for n in range(hyperparam['NUM_CLASS'])], index=range(1, hyperparam['EPOCHS']+1, 1))
    #     df_val = pd.DataFrame(columns=['class_' + str(n) for n in range(hyperparam['NUM_CLASS'])], index=range(1, hyperparam['EPOCHS']+1, 1))



# specify ImageNet mean and standard deviation #NB This should not change
# MEAN = (0.485, 0.456, 0.406)
# STD = (0.229, 0.224, 0.225)


#TODO: Arrange this script to have fully function config script 


# Classifier: __init__(self, numClasses, freeze=True,
                    #  model_dict={'repo_or_dir': 'pytorch/vision:v0.10.0',
                    #              'model': 'resnet18',
                    #              'pretrained': True,
                    #              'skip_validation': True})

# Metrics: __init__(self, num_class, prefix='', metric_ls=[])

# ziramutils functions: get_dataloader(dataset, batchSize, shuffle=True)
#                       train_val_split(dataset, valSplit=0.2)

# ZiramDataset: __init__(self, base_path: str, dataset: str, im_size: int, mean: tuple, std: tuple, filename: str, label: str, num_class: int,
                #  augment: bool = False, exlude_controls: bool = False)

# ZiramDataset(path=hyperparam['BASE_PATH'],
#                                 dataset='training_set', 
#                                 filename = hyperparam['FILENAME'],
#                                 label=hyperparam['LABEL'],
#                                 num_class=hyperparam['NUM_CLASS'],
#                                 im_size=hyperparam['IMAGE_SIZE'],
#                                 mean=config.MEAN, 
#                                 std=config.STD,
#                                 augment=True)



# specify training hyperparameters
# HYPERPARAMS = dict(
#     BASE_PATH = "/nfs/research/birney/users/fanny/medaka/ziram_analysis/dataset_ziram_CO4_no0",
#     FILENAME = 'CO4',
#     NUM_CLASS = 4,
#     NAME='removed_sev_0',
#     EXP_NAME='full_Ziram_dataset',
#     IMAGE_SIZE = 256,
#     BATCH_SIZE = 16,
#     PRED_BATCH_SIZE = 16,
#     EPOCHS = 5,
#     LABEL = 'severity_score',
#     LR = 0.001,
#     AUGMENT = True,
# )
 
# MLFLOW = True
# PLOT_AUROC = True

# # define paths to store training plot and trained model
# OUTPUT_PATH = '/hps/nobackup/birney/users/fanny/ziram'
# ML_OUTPUT = os.path.join(OUTPUT_PATH, "mlruns")
# OUTPUT_PATH = os.path.join(OUTPUT_PATH, "output_no0_tes")

# if not os.path.isdir(OUTPUT_PATH):
#     os.makedirs(OUTPUT_PATH)
# MODEL_PATH = os.path.join("output_no0", "model.pth")

def modif_params_argparse():
    parser = argparse.ArgumentParser(description="Modification of run parameters", argument_default=argparse.SUPPRESS)
    parser.add_argument("-P", "--BASE_PATH", type=str, help='')
    parser.add_argument("-F", "--FILENAME", type=str, help='')
    parser.add_argument("-L", "--LABEL", type=str, help='')
    parser.add_argument("-C", "--NUM_CLASS", type=int, help='')
    parser.add_argument("-N", "--NAME", type=str, help='')
    parser.add_argument("-E", "--EXP_NAME", type=str, help='')
    parser.add_argument("--IMAGE_SIZE", type=int, help='')
    parser.add_argument("--BATCH_SIZE", type=int, help='')
    parser.add_argument("--PRED_BATCH_SIZE", type=int, help='')
    parser.add_argument("--EPOCHS", type=int, help='')
    parser.add_argument("--LR", type=float)
    # parser.add_argument("--AUGMENT", type=bool, action='store_true')
    args = parser.parse_args()
    HYPERPARAMS.update(vars(args))


def extract_hyperparams(hyperparams):
    list_params = []
    if not isinstance(hyperparams['IMAGE_SIZE'], list): hyperparams['IMAGE_SIZE'] = [hyperparams['IMAGE_SIZE']]
    for im in hyperparams['IMAGE_SIZE']:
        if not isinstance(hyperparams['BATCH_SIZE'], list): hyperparams['BATCH_SIZE'] = [hyperparams['BATCH_SIZE']]
        for bat in hyperparams['BATCH_SIZE']:
            if not isinstance(hyperparams['LR'], list): hyperparams['LR'] = [hyperparams['LR']]
            for lr in hyperparams['LR']:
                list_params.append({'IMAGE_SIZE':im, 'BATCH_SIZE': bat, 'LR':lr})
    return list_params
            

def params_fromYAML(yaml_file):

    # Open the YAML param file
    with open(yaml_file, 'r') as y:
        yaml_dict = yaml.safe_load(''.join(y.readlines()))

    # Setting the params
    params=yaml_dict.get('CONFIG')
    
    # determine the device type 
    if params['DEVICE'] == 'cuda':
        if torch.cuda.is_available() == False: 
            print('[WARNING] CUDA not available using CPU instead')
            params['DEVICE'] = "cpu"
            device='cpu'
        else:
            params['DEVICE'] = torch.device("cuda")
            device='cuda'    

    # assert os.path.exists(params['OUTPUT_PATH']), 'OUTPUT_PATH non existant' 
    run_name = params['NAME']
    exp_name = params['EXP_NAME']


    hyperparams=yaml_dict.get('HYPERPARAMS')
    #TODO: To adapt in the case we are not in cluster slurm
    if hyperparams['mode'] == 'screen':
        jobID = int(os.environ['SLURM_ARRAY_TASK_ID'])-1
        hyperparams_list = extract_hyperparams(hyperparams)
        hyperparams = hyperparams_list[jobID]
        run_name += '_' + str(jobID)


    save_path = os.path.join(params['OUTPUT_PATH'], exp_name, run_name)
    n=0
    while os.path.isdir(save_path):
        n += 1
        save_path = os.path.join(params['OUTPUT_PATH'], exp_name, run_name) + '_' + str(n) 
    run_name += '_' + str(n)    
    os.makedirs(save_path)
    params.update({'OUTPUT_PATH':save_path})

    print(f'[INFO] Parameters of the run ({exp_name + "_" + run_name}):\n')
    print(pd.DataFrame(hyperparams, index=['Value']).T.to_markdown(), '\n')

    yaml_out = open(os.path.join(save_path, "run_info.yaml"), "w")
    yaml_dict['CONFIG'].update({'NAME': run_name, 
                                'DEVICE': device})
    yaml.dump(yaml_dict, yaml_out)
    yaml_out.close()
    
    # Adjusting the kwargs
    data_kwargs = yaml_dict.get('DATASET')
    model_kwargs=yaml_dict.get('MODEL')
    metrics_kwargs = yaml_dict.get('METRICS')
    data_kwargs.update(
        dict(im_size=hyperparams['IMAGE_SIZE'],
            batch_size=hyperparams['BATCH_SIZE'],
            mean=tuple(model_kwargs.pop('mean')), 
            std=tuple(model_kwargs.pop('std'))
            )
        ) 
    model_kwargs.update(dict(numClasses=data_kwargs['num_class']))

    return data_kwargs, model_kwargs, hyperparams, params, metrics_kwargs


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Training parameters")
#     parser.add_argument("-Y", "--yaml_file", type=str, help='Path of the YAML file containing the necessary information to run the training')
#     args = parser.parse_args()
#     params_fromYAML(yaml_file = args.yaml_file)
