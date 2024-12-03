# import the necessary packages
import torch
import os
import argparse
import sys
import yaml
import pandas as pd

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
    else:
        device='cpu'

    # assert os.path.exists(params['OUTPUT_PATH']), 'OUTPUT_PATH non existant' 
    run_name = params['NAME']
    exp_name = params['EXP_NAME']


    hyperparams=yaml_dict.get('HYPERPARAMS')
    #TODO: To adapt in the case we are not in cluster slurm
    if hyperparams['mode'] == 'screen':
        try:
            jobID = int(os.environ['SLURM_ARRAY_TASK_ID'])-1
            hyperparams_list = extract_hyperparams(hyperparams)
            hyperparams = hyperparams_list[jobID]
            run_name += '_' + str(jobID)
            yaml_dict['HYPERPARAMS'].update(hyperparams)
        except KeyError:
            print('The job has not been run as an array. JobID unavailable')


    save_path = os.path.join(params['OUTPUT_PATH'], exp_name, run_name)
    if os.path.isdir(save_path):
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
