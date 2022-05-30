"""
wrapper functions to use reinvent
"""


import subprocess
from train_agent import train_agent
import os
import pandas as pd
import joblib
import torch


MAX_CHEMS_PER_TRIAL = 5


# free gpu memory forcefully
# i.e., empty cashe did not work
def free_gpu():
    sp = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv"])
    ids = str(sp).split("\\n")
    ids = ids[1:-1]
    sp = subprocess.check_output(["kill", "-9", ids[0]])


def run_reinvent(fp,
                 arg_dict={
                     # "scoring_function":"polym_fp_similarity",
                     "scoring_function": "fp_similarity",
                     "n_steps": 300,
                 },
                 save_dir=None,
                 rein_dir='REINVENT/',
                 original_dir='../',
                 device="cuda:0",
                 log_experience=True,
                 ):

    if os.getcwd().split("/")[-1] != "REINVENT":
        os.chdir(rein_dir)

    fp = [str(i) for i in fp]
    fp = "".join(fp)

    arg_dict["scoring_function_kwargs"] = {"query_bit": fp}
    arg_dict["save_dir"] = save_dir
    arg_dict["device"] = device
    arg_dict["verbose"] = False
    arg_dict["log_experience"] = log_experience

    # run
    save_dir = train_agent(**arg_dict)

    # load data
    dqn_df = pd.read_csv(save_dir+"/memory", delimiter=" ")
    dqn_df = dqn_df.sort_values(by=["Score"], ascending=False)
    os.chdir(original_dir)

    return dqn_df


def run_reinvent_parallel(selected_fp_list,
                          arg_dict={
                              # "scoring_function":"polym_fp_similarity",
                              "scoring_function": "fp_similarity",
                              "n_steps": 300,
                          },
                          rein_dir='REINVENT/',
                          original_dir='../',
                          n_parallel=4,
                          n_chems=5,
                          gpu_num=1,
                          log_experience=True
                          ):

    def mini_dqn(arg):
        fp, i, gpu_id = arg[0], arg[1], arg[2]
        dqn_df = run_reinvent(fp,
                              arg_dict=arg_dict,
                              save_dir='data/results/run'+str(i),
                              rein_dir=rein_dir,
                              original_dir=original_dir,
                              device="cuda:"+str(gpu_id),
                              log_experience=log_experience)
        return dqn_df

    task_per_gpu = int(n_parallel/gpu_num)
    task_per_gpu
    gpu_ids = []
    for i in range(gpu_num):
        gpu_ids.extend([i]*task_per_gpu)
    for j in range(n_parallel-task_per_gpu*gpu_num):
        gpu_ids.append(i)
    print("gpus attributed: ", gpu_ids)

    job_arg = [(fp, i, gpu_id) for i, fp, gpu_id in zip(
        range(len(selected_fp_list)), selected_fp_list, gpu_ids)]
    dqn_df_list = joblib.Parallel(n_jobs=n_parallel)(
        joblib.delayed(mini_dqn)(arg) for arg in job_arg)

    for i, temp_df in enumerate(dqn_df_list):
        if i == 0:
            integ_df = temp_df[:n_chems]
        else:
            integ_df = pd.concat([integ_df, temp_df[:n_chems]])

    free_gpu()
    return integ_df
