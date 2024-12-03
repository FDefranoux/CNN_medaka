# import the necessary packages
from pyimagesearch.classifier import Larval_MLClassifier
from pyimagesearch.ziramutils import get_dataloader, ZiramDataset
from pyimagesearch.metrics import MetricRecorder
from torchvision import transforms
from torch.nn import Softmax
from torch import nn
import matplotlib.pyplot as plt
import argparse
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os
import torchmetrics
import yaml
from pyimagesearch.config import params_fromYAML
import datetime
import seaborn as sns


def main(training_output_dir, encodings_extract=False, plot_batch=False):

####################### PARAMS settings (to be put in config.py) #####################################
	test_kwargs, model_kwargs, hyperparams, params, metrics_kwargs = params_fromYAML(os.path.join(training_output_dir, 'run_info.yaml'))
	numClasses = test_kwargs.pop('num_class')
	run_name = params['NAME'] + '_test'
	state_dict_path = os.path.join(training_output_dir, 'model.pth')
	assert os.path.exists(state_dict_path), 'State_dict file not found'

	if params['MLFLOW']:
		import mlflow
		print(f'\n\n[INFO] Starting MLFLOW run\n', flush=True)
		mlflow.set_tracking_uri(params['MLFLOW_OUT'])
		mlflow.set_experiment(params['EXP_NAME'])
		mlflow.start_run(run_name=run_name)
		mlflow.log_artifact(os.path.join(training_output_dir, "run_info.yaml"))
		for key, item in hyperparams.items():
			mlflow.log_param(key, item)		


	# Recuperation of the mlflow infos
	# ABS_PATH = os.path.dirname(os.path.abspath(__file__))
	# if config.MLFLOW & (run_id != ''):
	# 	run_info = mlflow.get_run(run_id=run_id)
	# 	run_name = run_info.data.tags["mlflow.runName"]
	# 	# exp_name = run_info.data.tags["mlflow.expeName"]
	# 	run_params = run_info.data.params
	# 	mlflow.end_run()
	# 	#Recuperation of the state dict path
	# 	artifact_uri =run_info.info.artifact_uri.replace('flow-artifacts:', 'artifacts')
	# 	state_dict_path = '/hps/nobackup/birney/users/fanny/ziram/mlruns/1/2b1da5db7c374b67a9d181451ff06e4a/artifacts/best_auroc/state_dict.pth'
	# 	# os.path.join(artifact_uri, '.pth')[7:]
	# 	print('PATH', state_dict_path)
	# 	# config_path = os.path.join(artifact_uri, 'config.py')[7:]
	# 	# print(config_path)
	# 	# assert os.path.exists(config_path), 'config file not found'
	# 	# sys.path.insert(0, os.path.dirname(config_path))
	# else:
	# 	model_time = datetime.datetime.fromtimestamp(os.path.getmtime(config.MODEL_PATH))
	# 	yaml_time = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(config.OUTPUT_PATH, "run_info.yaml")))
	# 	print(model_time, yaml_time)
	# 	if (model_time.year, model_time.month, model_time.day, model_time.hour, model_time.minute) != (yaml_time.year, yaml_time.month, yaml_time.day, yaml_time.hour, yaml_time.minute): print('[WARNING] Model and YAML do not have same modification date')
	# 	yaml_file = open(os.path.join(config.OUTPUT_PATH, "run_info.yaml"),"r")
	# 	run_params = yaml.load(yaml_file, Loader=yaml.SafeLoader)
	# 	state_dict_path = config.MODEL_PATH
	# 	run_name = run_params.pop('NAME') 
	# 	exp_name = run_params.pop('EXP_NAME')
	

#################### Initialisation of evaluation ################################

	# create the test dataset 
	for key in ['validation_ratio', 'batch_size', 'shuffle']: del test_kwargs[key]
	Dataset = ZiramDataset(mode='test', **test_kwargs)
	assert int(Dataset.num_label) == int(numClasses), f'Numbers of labels ({Dataset.num_label}) not corresponding to class number ({numClasses})'
								

	# initialize the test data loader
	testLoader = get_dataloader(Dataset, 32)

	# build the custom model
	model = Larval_MLClassifier(**model_kwargs).to(params['DEVICE'])
	model.load_state_dict(torch.load(state_dict_path))  # load the model state
	
    # initialize loss function (criterion) and optimizer
	criterion = model.criterion()		# initialize the loss function
	
	# initialize test data loss
	test_metrics = MetricRecorder(num_class=numClasses,  prefix='Test', metric_ls=metrics_kwargs['EVALUATION'])
	# testCorrect, totalTestLoss  = 0, 0
	# testAuroc = torchmetrics.AUROC(task="multiclass", num_classes=run_params['NUM_CLASS'])
	# soft = Softmax()

	# switch off autograd
	res_df = pd.DataFrame()

	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		tqdm_object = tqdm(testLoader, total=len(testLoader), desc='Evaluation', colour='magenta')

		# loop over the testing set
		# tot_encoding, tot_logits, tot_coef = np.empty([run_params['PRED_BATCH_SIZE'], 1000]), np.empty(shape=[run_params['PRED_BATCH_SIZE'], run_params['NUM_CLASS']]), np.empty([run_params['PRED_BATCH_SIZE'], 1])
		tot_encoding = np.empty([0, 1000])
		tot_pth = []
		n = 0
		for batch in tqdm_object:
			n += 1 

			# send the input to the device
			image_batch, target = (batch['image'].to(params['DEVICE']), batch['label_' + str(model_kwargs["mode"])].to(params['DEVICE']))
			if  model_kwargs['mode'] == 'regression':
				target = torch.unsqueeze(target, -1).to(params['DEVICE'])
         	
			# Compute loss
			out_dict = model(image_batch)
			# make the predictions and calculate the evaluation loss
			logits = out_dict['logits'].to(params['DEVICE'])
			if  model_kwargs['mode'] == 'regression':
				crit_loss = criterion(logits, target.type(torch.FloatTensor)).to(params['DEVICE'])
			else:
				crit_loss = criterion(logits, target).to(params['DEVICE'])


			results_batch = pd.DataFrame({key:item for key, item in batch.items() if key in ['img_path', 'label_class', 'label_reg']})
			results_batch[[ 'logits_' + str(n) for n in range(numClasses)]] = logits
			res_df = res_df.append(results_batch)

			if encodings_extract:
				#### TEST encodings
				batch_encoding = out_dict['encoding']
				tot_encoding = np.append(tot_encoding, batch_encoding, axis=0)
				tot_pth += list(batch['img_path'])

			# classif_loss = criterion_classif(logits_class, target_class).type(torch.FloatTensor)
			# regression_loss = criterion_regression(coefficient_reg, target_reg.type(torch.FloatTensor)) 
			# crit_loss = classif_loss + regression_loss
			test_metrics.metric_update(logits=logits, targets=target, loss=crit_loss)
			# totalTestLoss += loss.item()
			# output logits through the softmax layer to get output



			# predictions, and calculate the number of correct predictions
			# testCorrect += (logit.argmax(dim=-1) == target_batch).sum().item()
			# testAuroc.update(torch.nn.functional.softmax(logit, dim=-1), target_batch)
			# testAuroc_value = testAuroc.compute().cpu().numpy()		
			# tqdm_object.set_postfix(TestAuroc=testAuroc_value.mean())
			# mlflow.log_metric(key='totalTestLoss', value=totalTestLoss, step=batch)
			# mlflow.log_metric(key='testCorrect', value=testCorrect, step=batch)
			# for n in range(run_params.NUM_CLASS): 
			# 	mlflow.log_metric(key=f'testAuroc_{n}', value=float(testAuroc_value[n]), step=batch)

			mean_test_metrics = test_metrics.compute_array()
			print(mean_test_metrics)
			if params['MLFLOW']:
				test_metrics.save_mlflow(epoch=n, mean=False)

	print('\n\n### Test set results')	
	print(res_df.shape)
	res_df.to_csv(os.path.join(training_output_dir, 'Testset_results.csv'), index=False)

	fig = sns.heatmap(test_metrics.dict_array['Conf_matrix'], linewidth=0.5, annot=True, cbar=False, cmap='Spectral')
	try:
		fig.savefig(os.path.join(training_output_dir, 'ConfusionMatrix_testset.jpg'), bbox_inches='tight', dpi=100)
	except AttributeError:
		fig.get_figure().savefig(os.path.join(training_output_dir, 'ConfusionMatrix_testset.jpg'), bbox_inches='tight', dpi=100)

	if encodings_extract:
		print('\n\n### Models ENCODINGs')		
		encoding_df = pd.DataFrame(tot_encoding, index=tot_pth)
		encoding_df.to_csv(os.path.join(training_output_dir, 'Testset_encodings.csv'))

	# print test data accuracy
	# print('\n[INFO] Final Accuracy', testCorrect/len(testDataset), 'Auroc', testAuroc_value.mean())
	
	if plot_batch:
		# initialize iterable variable
		sweeper = iter(testLoader)
		
		# grab a batch of test data
		batch = next(sweeper)
		(images, labels) = (batch['image'], batch['label'])
		
		# initialize a figure
		from math import ceil
		fig, axs = plt.subplots(ceil(len(images)/6), 6, figsize=(50,50))

		# calculate the inverse mean and standard deviation to define our denormalization transform
		invMean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
		invStd = [1/s for s in config.STD]
		deNormalize = transforms.Normalize(mean=invMean, std=invStd)

		# switch off autograd
		with torch.no_grad():
			# send the images to the device
			images = images.to(config.DEVICE)
			# make the predictions
			preds = model(images)
			# loop over all the batch
			for i, (image, ax) in enumerate(zip(images, axs.flatten())):
				# grab the image, de-normalize it, scale the raw pixel
				# intensities to the range [0, 255], and change the channel
				# ordering from channels first tp channels last
				# image = images[i]
				image = deNormalize(image).cpu().numpy()
		

				# grab the ground truth label
				idx = labels[i].cpu().numpy()

				# grab the predicted label
				pred = preds[i].argmax().cpu().numpy()
				result = idx == pred
				# add the results and image to the plot
				info = "Result: {} - (Ground Truth: {}, Predicted: {})".format(result, idx, pred)
				# print(info)
				image_perm = (image.permute((1, 2, 0)) / torch.max(torch.abs(image))+1)/2
				ax.imshow(image_perm, cmap='Greys')
				ax.title.set_text(info)
				# ax.axis("off")
			[axi.axis('off') for axi in axs.ravel()]
		
		# # show the plot
		plt.tight_layout()
		# fig.suptitle('Accuracy='+ str(testCorrect/len(testDataset)) + '--  Auroc='+ str(round(float(testAuroc_value.mean()), 2)))
		fig.savefig(os.path.join(training_output_dir, 'result_test_cnn_new.png'))

	print('[INFO] Resuts saved !')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Evaluation of CNN run of test dataset", argument_default=argparse.SUPPRESS)
	# parser.add_argument("--run_id", help="run-id from which we need to recuperate the training")
	parser.add_argument("-o", "--training_output_dir", help="Output directory of the training")
	parser.add_argument("-p", "--plot_batch", help="To print out and save plots of a batch results", action="store_true")
	parser.add_argument("-E", "--encodings_extract", help="To extract the encodings of the test set", action="store_true")
	args = parser.parse_args()
	main(**vars(args))
