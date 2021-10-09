"""Read in event file written by tensorboard and perform post hoc comparison. """
import os
from glob import glob
from pathlib import Path
import torch
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml
rootdir = Path(r"E:\Cluster_Backup\SimCLR-runs")
expdir_col = ["proj256_eval_sal_new_T0.01_Oct06_19-02-51",
        "proj256_eval_sal_new_T0.1_Oct06_19-02-51",
        "proj256_eval_sal_new_T0.3_Oct06_19-10-25",
        "proj256_eval_sal_new_T0.7_Oct08_08-44-40",
        "proj256_eval_sal_new_T1.0_Oct06_19-02-50",
        "proj256_eval_sal_new_T1.5_Oct08_08-44-40",
        "proj256_eval_sal_new_T2.5_Oct08_08-44-40",
        "proj256_eval_sal_new_T3.0_Oct06_19-10-25",
        "proj256_eval_sal_new_T4.5_Oct08_08-49-50",
        "proj256_eval_sal_new_T10.0_Oct06_19-10-25",
        "proj256_eval_sal_new_T30.0_Oct06_19-10-25",
        "proj256_eval_sal_new_T100.0_Oct06_19-05-06",
        "proj256_eval_sal_new_flat_Oct06_19-02-51", # use the flat saliency map as substitute.keep the sampling mechanism.
        ]

T_arr = []
train_acc_arr = np.ones((22, len(expdir_col),)) * np.nan
test_acc_arr = np.ones((22, len(expdir_col),)) * np.nan
simclr_acc_arr = np.ones((392, len(expdir_col),)) * np.nan
for ei, expdir in enumerate(expdir_col):
    expfp = rootdir/expdir
    fns = glob(str(expfp/"events.out.tfevents.*"))
    assert len(fns) == 1
    event_acc = EventAccumulator(str(expfp))
    event_acc.Reload()
    step2epc_map = {step: epc for _, step, epc in event_acc.Scalars('epoch')}
    step2epc_map[-1] = -1
    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    _, eval_step_nums, test_acc_val = zip(*event_acc.Scalars('eval/test_acc'))
    _, eval_step_nums, train_acc_val = zip(*event_acc.Scalars('eval/train_acc'))
    _, train_step_nums, simclr_acc_val = zip(*event_acc.Scalars('acc/top1'))
    epocs = np.array(eval_step_nums)//390
    cfgargs = yaml.load(open(expfp / "config.yml", 'r'), Loader=yaml.Loader)
    temperature = cfgargs.crop_temperature
    sal_control = cfgargs.sal_control
    T_arr.append(temperature)
    train_acc_arr[:len(train_acc_val), ei] = np.array(train_acc_val)
    test_acc_arr[:len(test_acc_val), ei] = np.array(test_acc_val)
    simclr_acc_arr[:len(simclr_acc_val), ei] = np.array(simclr_acc_val)

#%%
import matplotlib.pylab as plt
epoc_id = 1
figh = plt.figure(figsize=(4,5))
plt.plot(T_arr[:-1], train_acc_arr[epoc_id,:-1], label="Train Set")
plt.plot(T_arr[:-1], test_acc_arr[epoc_id,:-1], label="Test Set")
plt.hlines(train_acc_arr[epoc_id, -1],0,100,color="darkblue",linestyles=":",
           label="Train (Uniform Sampling)")
plt.hlines(test_acc_arr[epoc_id, -1],0,100,color="red",linestyles=":",
           label="Test (Uniform Sampling)")
plt.semilogx()
plt.xlim([0,100])
plt.legend()
plt.xlabel("Sampling Temperature")
plt.ylabel("Linear Eval Accuracy")
plt.title("Visual Repr Evaluation")
plt.show()
#%%
step_id = -20
figh = plt.figure(figsize=(4,5))
plt.plot(T_arr[:-1], simclr_acc_arr[step_id,:-1], label="Simclr Acc")
plt.hlines(simclr_acc_arr[step_id, -1],0,100,color="darkblue",linestyles=":",
           label="Simclr Acc (Uniform Sampling)")
plt.semilogx()
plt.xlim([0,100])
plt.legend()
plt.xlabel("Sampling Temperature")
plt.ylabel("Unlabeled Set Simclr Accuracy")
plt.show()
# Show all tags in the log file
# print(event_acc.Tags())
# 'scalars': ['eval/train_loss', 'eval/train_acc', 'eval/test_loss', 'eval/test_acc', 'epoch', 'loss', 'acc/top1', 'acc/top5', 'learning_rate'],
#%%
proj256_eval_magnif_bsl_Oct07_05-11-35
proj256_eval_magnif_bsl_Oct07_19-46-29
proj256_eval_magnif_cvr_0_01-1_50_Oct07_05-06-53
proj256_eval_magnif_cvr_0_05-0_35_Oct07_05-06-55
proj256_eval_magnif_cvr_0_01-0_35_Oct07_05-06-57
proj256_eval_magnif_cvr_0_05-0_70_Oct07_05-11-35
proj256_eval_magnif_cvr_0_05-0_70_Oct07_19-46-29
proj256_eval_magnif_cvr_0_01-0_35_Oct07_19-46-40
proj256_eval_magnif_cvr_0_05-0_35_Oct07_19-46-40
proj256_eval_magnif_cvr_0_01-1_50_Oct07_19-46-40
proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-24
proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-26
proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-31
proj256_eval_magnif_exp_cvr_0_05-1_00_slp_1_50_Oct07_19-21-47
proj256_eval_magnif_exp_cvr_0_05-1_00_slp_0_75-3_00_Oct07_19-38-20
proj256_eval_magnif_exp_cvr_0_05-1_00_slp_0_75-3_00_Oct07_19-22-11
proj256_eval_magnif_exp_cvr_0_05-0_50_slp_1_50_Oct07_19-25-55
proj256_eval_magnif_exp_cvr_0_05-0_50_slp_0_75-3_00_Oct07_19-22-28

#%% Baseline Distribution Fixed Seed
expdirs = ["proj256_eval_magnif_bsl_Oct07_05-11-35",
        "proj256_eval_magnif_bsl_Oct07_19-46-29",
        "proj256_eval_magnif_bsl_Oct08_07-44-26",
        "proj256_eval_magnif_bsl_Oct08_07-44-27",
        "proj256_eval_magnif_bsl_Oct08_07-44-30",
        "proj256_eval_magnif_bsl_Oct08_07-45-39",
        "proj256_eval_magnif_bsl_Oct08_07-45-41",
        "proj256_eval_magnif_bsl_Oct08_07-45-43",
        "proj256_eval_magnif_bsl_Oct08_07-45-48",
        ]

#%% 
expdirs = [
"proj256_eval_magnif_salmap_T0_3_cvr_0_01-0_35_Oct08_07-19-53",
"proj256_eval_magnif_salmap_T0_7_cvr_0_01-0_35_Oct08_07-18-58",
"proj256_eval_magnif_salmap_T1_0_cvr_0_01-0_35_Oct08_07-18-59",
"proj256_eval_magnif_salmap_T1_5_cvr_0_01-0_35_Oct08_07-18-59",
"proj256_eval_magnif_salmap_T10_0_cvr_0_01-0_35_Oct08_07-17-52",
]




#%%
quad_magnif_expdirs = ["proj256_eval_magnif_cvr_0_01-1_50_Oct07_05-06-53",
                     "proj256_eval_magnif_cvr_0_01-1_50_Oct07_19-46-40",
                     "proj256_eval_magnif_cvr_0_05-0_70_Oct07_05-11-35",
                     "proj256_eval_magnif_cvr_0_05-0_70_Oct07_19-46-29",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct07_05-06-55",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct07_19-46-40",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-24",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-24-SPLIT",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-26",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-31",
                     "proj256_eval_magnif_cvr_0_01-0_35_Oct07_19-46-40",
                     "proj256_eval_magnif_cvr_0_01-0_35_Oct07_05-06-57",] 

expdirs = quad_magnif_expdirs
param_list = []
train_acc_arr = np.ones((22, len(expdirs)+1,)) * np.nan
test_acc_arr = np.ones((22, len(expdirs)+1,)) * np.nan
simclr_acc_arr = np.ones((392, len(expdirs)+1,)) * np.nan
for ei, expdir in enumerate(expdirs):
    expfp = rootdir/expdir
    fns = glob(str(expfp/"events.out.tfevents.*"))
    assert len(fns) == 1
    event_acc = EventAccumulator(str(expfp))
    event_acc.Reload()
    step2epc_map = {step: epc for _, step, epc in event_acc.Scalars('epoch')}
    step2epc_map[-1] = -1
    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    _, eval_step_test, test_acc_val = zip(*event_acc.Scalars('eval/test_acc'))
    _, eval_step_train, train_acc_val = zip(*event_acc.Scalars('eval/train_acc'))
    _, train_step_nums, simclr_acc_val = zip(*event_acc.Scalars('acc/top1'))
    epocs = np.array(eval_step_test)//390
    if expdir == 'proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-24':
        test_acc_val = test_acc_val[::2]
        train_acc_val = train_acc_val[::2]
        eval_step_train = eval_step_train[::2]
        eval_step_test = eval_step_test[::2]
        train_step_nums = train_step_nums[::2]
        simclr_acc_val = simclr_acc_val[::2]
    # if expdir == 'proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-31':
    #     test_acc_val = test_acc_val[::3]
    #     train_acc_val = train_acc_val[::3]
    #     eval_step_train = eval_step_train[::3]
    #     eval_step_test = eval_step_test[::3]
    #     train_step_nums = train_step_nums[::3]
    #     simclr_acc_val = simclr_acc_val[::3]
    cfgargs = yaml.load(open(expfp / "config.yml", 'r'), Loader=yaml.Loader)
    cover_ratio = cfgargs.cover_ratio
    param_list.append(cover_ratio)
    train_acc_arr[:len(train_acc_val), ei] = np.array(train_acc_val)
    test_acc_arr[:len(test_acc_val), ei] = np.array(test_acc_val)
    simclr_acc_arr[:len(simclr_acc_val), ei] = np.array(simclr_acc_val)
    print("cover_ratio [%.2f, %.2f] trainACC %.4f  testACC %.4f  simclrACC %.4f"%(*cover_ratio, train_acc_arr[-2,ei], test_acc_arr[-2,ei], simclr_acc_arr[-3,ei]))

#%%
SIMCLR_LEN = 390
EVAL_LEN = 22
INVALIDTIME = -99999
def split_record(timestep, value, standardL=EVAL_LEN):
    """ Split events of different threads in a same record file """
    Tnum = len(timestep)
    threadN = np.ceil(Tnum / standardL,).astype(np.int)
    thread_mask = []
    cnt_per_thread = [0 for _ in range(threadN)]
    thread_pointer = 0 
    last_event = None
    for i, T in enumerate(timestep):
        if T == last_event:
            thread_pointer += 1
        else:
            thread_pointer = 0 
        thread_mask.append(thread_pointer) 
        cnt_per_thread[thread_pointer] += 1 
        last_event = T

    assert len(thread_mask) == len(timestep) == len(value)
    timestep = np.array(timestep)
    value = np.array(value)
    thread_mask = np.array(thread_mask)
    time_threads = [timestep[thread_mask==i]  for  i  in  range(threadN)]
    val_threads = [value[thread_mask==i]  for  i  in  range(threadN)]
    return [np.concatenate((time_arr,
                    INVALIDTIME * np.ones(standardL-len(time_arr), dtype=time_arr.dtype)))
                    for  time_arr  in  time_threads], \
           [np.concatenate((val_arr,
                    np.nan * np.ones(standardL-len(val_arr), dtype=val_arr.dtype)))
                    for  val_arr  in  val_threads]

import pandas as pd
keys = ["cover_ratio"]
def load_format_exps(expdirs, cfgkeys=["cover_ratio"]):
    train_acc_col = [] # 22
    test_acc_col = []  # 22
    simclr_acc_col = [] # 390
    param_list = []
    expnm_list = []
    for ei, expdir in enumerate(expdirs):
        expfp = rootdir/expdir
        fns = glob(str(expfp/"events.out.tfevents.*"))
        assert len(fns) == 1
        event_acc = EventAccumulator(str(expfp))
        event_acc.Reload()
        cfgargs = yaml.load(open(expfp / "config.yml", 'r'), Loader=yaml.Loader)
        # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
        _, eval_step_test, test_acc_val = zip(*event_acc.Scalars('eval/test_acc'))
        _, eval_step_train, train_acc_val = zip(*event_acc.Scalars('eval/train_acc'))
        _, train_step_nums, simclr_acc_val = zip(*event_acc.Scalars('acc/top1'))
        # step2epc_map = {step: epc for _, step, epc in event_acc.Scalars('epoch')}
        # step2epc_map[-1] = -1
        # Split threads of the same experiments
        eval_step_test_thrs, test_acc_val_thrs = split_record(eval_step_test, test_acc_val, EVAL_LEN)
        eval_step_train_thrs, train_acc_val_thrs = split_record(eval_step_train, train_acc_val, EVAL_LEN)
        train_step_nums_thrs, simclr_acc_val_thrs = split_record(train_step_nums, simclr_acc_val, SIMCLR_LEN)
        thread_num = len(eval_step_test_thrs)
        assert len(test_acc_val_thrs) == len(train_acc_val_thrs) == len(simclr_acc_val_thrs)
        train_acc_col.extend(train_acc_val_thrs)
        test_acc_col.extend(test_acc_val_thrs)
        simclr_acc_col.extend(simclr_acc_val_thrs)
        param_list.extend([{k:cfgargs.__getattribute__(k) for k in cfgkeys}] * thread_num)
        expnm_list.extend([expdir] * thread_num)

    eval_timestep = np.array([-1, *range(1,100,5), 100])
    simclr_timestep = np.array([*range(0,39000,100)])
    assert(len(eval_timestep) == EVAL_LEN)
    assert(len(simclr_timestep) == SIMCLR_LEN)
    train_acc_arr = np.array(train_acc_col)
    test_acc_arr = np.array(test_acc_col)
    simclr_acc_arr = np.array(simclr_acc_col)
    param_table = pd.DataFrame(param_list)
    param_table.index = expnm_list
    return  train_acc_arr, test_acc_arr, simclr_acc_arr, \
            eval_timestep, simclr_timestep, param_table

quad_magnif_expdirs = ["proj256_eval_magnif_cvr_0_01-1_50_Oct07_05-06-53",
                     "proj256_eval_magnif_cvr_0_01-1_50_Oct07_19-46-40",
                     "proj256_eval_magnif_cvr_0_05-0_70_Oct07_05-11-35",
                     "proj256_eval_magnif_cvr_0_05-0_70_Oct07_19-46-29",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct07_05-06-55",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct07_19-46-40",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-24",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-24-SPLIT",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-26",
                     "proj256_eval_magnif_cvr_0_05-0_35_Oct08_02-19-31",
                     "proj256_eval_magnif_cvr_0_01-0_35_Oct07_19-46-40",
                     "proj256_eval_magnif_cvr_0_01-0_35_Oct07_05-06-57",]

train_acc_arr, test_acc_arr, simclr_acc_arr, eval_timestep, simclr_timestep, \
    param_table = load_format_exps(expdirs, cfgkeys=["cover_ratio"])
#%%
for ei in range(param_table.shape[0]):
    print("cover_ratio [%.2f, %.2f] trainACC %.4f  testACC %.4f  simclrACC %.4f"%(*param_table.cover_ratio[ei], train_acc_arr[ei,-2], test_acc_arr[ei,-2], simclr_acc_arr[ei,-2]))
