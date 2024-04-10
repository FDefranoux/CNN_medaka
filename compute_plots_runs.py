import os
import sys
import pandas as pd
import mlflow
import argparse

def from_runlist(run_ids=[], metric_ls=[], tracking_uri='./mlruns'):
    final_metrics = pd.DataFrame()
    for run in run_ids: 
        
        df_metric = pd.DataFrame()
        # run_info = mlflow.get_run(run_id=str(run))
        # metrics_avail = list(run_info.data.metrics.keys())
        client = mlflow.tracking.MlflowClient(tracking_uri)
        # print(client.get_metric_history(run))

        # sys.exit()
        for metric in metric_ls:
            try:
                metric_info = client.get_metric_history(run, key=metric[8:])
                df_metric[metric] = pd.DataFrame([{'step': x.step, metric: x.value } for x in metric_info]).set_index('step')[metric]
            except:
                print(metric, ' not found')
        df_metric['run_id'] = run
        final_metrics = pd.concat([final_metrics, df_metric], axis=0)

    assert final_metrics.shape[0] != 0, "No metrics found"

    melt = final_metrics.reset_index().melt(id_vars=['run_id', 'step'])
    melt['set'] =  melt['variable'].replace(regex={'.*' + val[:-3] + '.*' : val for val in ['training', 'valid ', 'random   ']})
    melt['metric'] = melt[['variable']].replace(regex={'.*' + val + '.*' : val for val in ['Auroc', 'Loss', 'Acc']})
    melt['class'] = melt['variable'].str.split('_', expand=True)[1]

    return melt
            

def from_experiment_id(expe_id='', metric_regex='valAuroc'):
    df_runs = mlflow.search_runs(experiment_ids=list(expe_id))
    metrics_avail = [val for val in df_runs.columns if 'metrics.' in val]
    best_runs = df_runs.set_index('run_id').filter(regex=metric_regex).idxmax()
    # df_info = pd.DataFrame()
    # # print('\nclient run info1', client.list_run_infos(expe_id, run_view_type=1))
    # for n, run in enumerate(client.list_run_infos(expe_id, run_view_type=1)):
    #     df_info = df_info.append(pd.DataFrame({str(l[0]) : str(l[1]) for l in run }, index=[n]))
    # assert df_info.shape[0] != 0, 'No run found for this experiment ID !'
    
    # for col in df_info.filter(regex='.*time'):
    #     df_info[col] = pd.to_datetime(df_info[col], dayfirst=True, unit='ms', errors='ignore')
    # df_info.set_index(['experiment_id', 'run_id'], inplace=True)
    # # list_info = {n : {str(l[0]) : str(l[1])} for n, el in enumerate(client.list_run_infos(expe_id, run_view_type=1)) for l in el }
    # # print(list_info)
    # # df_info = pd.DataFrame(list_info)

    # ## TODO choose a way t plot if not in interactive mode. Which run do you plot ?
    # print(df_info[['start_time', 'end_time', 'status', 'user_id', 'lifecycle_stage', 'run_uuid']])

def main(tracking_uri='./mlruns', expe_id='', run_ls='', metric_ls='', interactive=True):
    client = mlflow.tracking.MlflowClient(tracking_uri)
    run_ls = run_ls.split()

    print(tracking_uri)
    if 'mlruns' not in tracking_uri: tracking_uri = os.path.join(tracking_uri, 'mlruns')
    assert os.path.isdir(tracking_uri), 'No mlruns folder in this URI'
    
    if (interactive == True) & (expe_id == None) & (run_ls == None):
        expe_id = input('Which experiment ID ?')

    df_runs = mlflow.search_runs(experiment_ids=list(expe_id))  # == in bash write  `mlflow experiments csv -x $expe_id`
    metric_cols = list(df_runs.filter(like='metrics.').dropna(how='all').columns)
    if (expe_id != None) & (run_ls == []):
        # 2. Give summary info to choose runs contained in the experiment
        if df_runs[df_runs['status'] == 'COMPLETED'].dropna(subset=metric_cols, how='all').shape[0] == 0: print('[WARN] No COMPLETED run found for this experiment ID !')
        select_cols = ["experiment_id","run_id","status","start_time","end_time", 'tags.mlflow.runName']  + list(df_runs.filter(like='params.').columns)
        
        print('\nGeneral info of the COMPLETED runs of this experiment\n', 
              df_runs[df_runs['status'] == 'COMPLETED'].filter(items=select_cols).set_index(['experiment_id', 'run_id', 'tags.mlflow.runName']).to_markdown())
        print('\nGeneral measures of the COMPLETED runs of this experiment\n', 
              df_runs.dropna(subset=metric_cols, how='all').set_index(['experiment_id', 'run_id', 'tags.mlflow.runName']).filter(items=metric_cols).to_markdown())
        print('\nBest run for each metric \n',  df_runs.set_index('run_id').filter(like='metrics.').idxmax().sort_index().to_markdown())
              
        # Input for several run id 
        run_ls = list(map(str, input("\nEnter a list of runs: ").split()))
    
    if (metric_ls == '') and (interactive == True):
        metric_ls = input("\nEnter a list of metrics for analysis (type `all` for all): ")
    if ' ' in metric_ls:
        metric_ls = metric_ls.split()
        metric_ls = ['metric.'+ metric for metric in metric_ls if 'metric.'+ metric in metric_cols]
        assert len(metric_ls) != 0, 'Metrics not found'
    elif metric_ls in metric_cols:
        metric_ls = [metric_ls]
    elif '*' in metric_ls:
        metric_ls = list(df_runs.filter(regex=metric_ls).dropna(how='all').columns)
    else:
        print('metric cols', metric_cols) 
        metric_ls = metric_cols
        
    if run_ls != None:
        # 3. compute the plots
        import seaborn as sns 
        metrics = from_runlist(run_ids=run_ls, metric_ls=metric_ls, tracking_uri=tracking_uri)
        run_name_dict = df_runs.set_index('run_id')['tags.mlflow.runName'].to_dict()
        metrics['run_name'] = metrics['run_id'].map(run_name_dict)
        metrics.dropna(how='all', inplace=True)
        metrics.dropna(how='all', inplace=True, axis=1)

        g = sns.relplot(data=metrics, x='step', y='value', row='metric', hue='set', kind='line', style='run_id', col='class')
        g.savefig('run_id.png')

        g = sns.relplot(data=metrics, x='step', y='value', row='metric', hue='set', kind='line', style='run_name', col='class')
        g.savefig('run_name.png')

    else: 
        print('No choice made here !')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run tracking")
    parser.add_argument("-t", "--tracking_uri", type=str, default='./mlruns', 
                        help='Tracking URI containing the mlrun folder where the tracking has been recorded')
    parser.add_argument("-i", "--interactive",  action='store_true',
                        help='Interactive option is available if you do not have run-ids or experiments id')
    parser.add_argument("-e", "--expe_id", type=str, default='',
                        help='You might have to specify the tracking URI, if working folder does not contains the mlruns folder')
    parser.add_argument("-r", "--run_ls", type=str, default='',
                        help='You might have to specify the tracking URI, if working folder does not contains the mlruns folder')
    parser.add_argument("-m", "--metric_ls", type=str, default='',
                        help='You might have to specify the tracking URI, if working folder does not contains the mlruns folder')
    args = parser.parse_args()
    main(**vars(args))