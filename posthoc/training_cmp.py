"""Read in event file written by tensorboard and perform post hoc comparison. """
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
event_acc = EventAccumulator(r'E:\Cluster_Backup\SimCLR-runs\proj256_eval_sal_new_T0.3_Oct05_07-10-22')
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
w_times, step_nums, vals = zip(*event_acc.Scalars('eval/test_acc'))