# import the necessary packages
import torch
import os
import argparse

# determine the device type 
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# specify ImageNet mean and standard deviation #NB This should not change
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


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
HYPERPARAMS = dict(
    BASE_PATH = "/nfs/research/birney/users/fanny/medaka/ziram_analysis/dataset_ziram_CO4_no0",
    FILENAME = 'CO4',
    NUM_CLASS = 4,
    NAME='removed_sev_0',
    EXP_NAME='full_Ziram_dataset',
    IMAGE_SIZE = 256,
    BATCH_SIZE = 16,
    PRED_BATCH_SIZE = 16,
    EPOCHS = 50,
    LABEL = 'severity_score',
    LR = 0.001,
    AUGMENT = True,
)
 
MLFLOW = True
PLOT_AUROC = True

# define paths to store training plot and trained model
OUTPUT_PATH = '/hps/nobackup/birney/users/fanny/ziram'
ML_OUTPUT = os.path.join(OUTPUT_PATH, "mlruns")
OUTPUT_PATH = os.path.join(OUTPUT_PATH, "output_no0_tes")

if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
MODEL_PATH = os.path.join("output_no0", "model.pth")

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



    
