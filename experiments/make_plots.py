import numpy as np
import os
import seaborn as sns
import sys

from matplotlib import pyplot as plt


#
problem = "animation"
batch_size = 3

if problem == "levy":
    title = "levy"
    opt_val = 0.0
elif problem == "hartmann":
    title = "hartmann"
    n_iter = 300
    n_trials = 50
    opt_val = 3.32237
elif problem == "rosenbrock":
    title = "rosenbrock"
    opt_val = 0.0
elif problem == "ackley":
    title = "ackley"
    n_iter = 300
    n_trials = 50
    opt_val = 0.0
elif problem == "michalewicz":
    title = "michalewicz"
    opt_val = 4.6876
elif problem == "carcab":
    title = "carcab"
    n_iter = 300
    n_trials = 50
    opt_val = -1.8101
elif problem == "alpine1":
    title = "alpine1"
    n_iter = 300
    n_trials = 50
    opt_val = 0.0
elif problem == "sushi":
    title = "sushi"
    n_iter = 250
    n_trials = 50
    opt_val = 0.0
elif problem == "animation":
    title = "animation"
    # opt_val = 3.1742  # 3.5064
    # opt_val = 3.4341
    opt_val = 13.9267  # 0.5240

n_trials = 50

if batch_size == 2:
    n_iter = 200
    problem_filename = problem
    algos = ["Random", "NEI", "TS", "EI", "EMOV"]
    labels = ["Random", "qNEI", "qTS", "qEI", "qEUBO"]
    colors = ["dimgrey", "orange", "red", "green", "blue"]
else:
    n_iter = 100
    problem_filename = problem  # + "_" + str(batch_size)
    algos = ["EI", "TS"]
    labels = ["Standard", "Composite"]
    colors = ["green", "red"]
    # algos = ["TS", "EI", "EMOV"]
    # labels = ["qTS", "qEI", "qEUBO", "qNEI", "Random", "A-EUBO"]
    # colors = ["red", "green", "blue", "orange", "dimgrey", "yellow"]
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
problem_results_dir = script_dir + "/results/" + problem + "/"


#
sns.set()
sns.set_palette("bright")
# get_ipython().run_line_magic('matplotlib', 'inline')
fontsize = 20
plt.rc("xtick", labelsize=fontsize)
plt.rc("ytick", labelsize=fontsize)

#
in_sample_flags = [True, False]
log_regret_flags = [True, False]

for in_sample in in_sample_flags:
    for log_regret in log_regret_flags:
        fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        for a, algo in enumerate(algos):
            algo += "_" + str(batch_size)
            print(algo)
            n_iter_algo = n_iter[a] if isinstance(n_iter, list) else n_iter
            algo_results_dir = problem_results_dir + algo + "/"

            best_obj_vals_all_trials = np.zeros((n_trials, n_iter_algo + 1))

            for trial in range(1, n_trials + 1):
                print(trial)
                if in_sample:
                    file_tag = "max_obj_vals_within_queries_"
                else:
                    file_tag = "obj_vals_at_max_post_mean_"
                best_obj_vals_all_trials[trial - 1, :] = np.loadtxt(
                    algo_results_dir + file_tag + str(trial) + ".txt"
                )[: n_iter_algo + 1]

            print(np.max(best_obj_vals_all_trials))
            if log_regret:
                best_obj_vals_all_trials = np.log10(opt_val - best_obj_vals_all_trials)

            mean = best_obj_vals_all_trials.mean(axis=0)
            if algo == "EMOV_3":
                for i in range(len(mean)):
                    mean[i] = mean[i] - 0.00 * i
            sem = best_obj_vals_all_trials.std(axis=0) / np.sqrt(n_trials)

            ax1.plot(range(0, n_iter_algo + 1), mean, label=labels[a], color=colors[a])
            if algo in ["Random_2", "NEI_2"]:
                ax1.fill_between(
                    range(0, n_iter_algo + 1),
                    mean - 1.96 * sem,
                    mean + 1.96 * sem,
                    alpha=0.15,
                    color=colors[a],
                )
            else:
                ax1.fill_between(
                    range(0, n_iter_algo + 1),
                    mean - 1.96 * sem,
                    mean + 1.96 * sem,
                    alpha=0.15,
                    color=colors[a],
                )

        problem = "dropwave"
        if in_sample:
            title = problem + " (in-sample)"
        else:
            title = problem

        if log_regret:
            ax1.set_xlabel("number of queries", fontsize=fontsize)
            ax1.set_ylabel("log$_{10}$(regret)", fontsize=fontsize)
            ax1.legend(loc="upper right", fontsize=fontsize - 5)
            ax1.set_title(title, fontsize=fontsize)
            if in_sample:
                fig1.savefig(
                    problem_results_dir + problem_filename + "_is_lr.pdf",
                    bbox_inches="tight",
                )
            else:
                fig1.savefig(
                    problem_results_dir + problem_filename + "_lr.pdf",
                    bbox_inches="tight",
                )
            # fig1.savefig(problem_results_dir + problem + "_lr.pdf", bbox_inches="tight")
            # fig1.savefig(problem_results_dir + problem + "_lr.eps", bbox_inches="tight")
        else:
            ax1.set_xlabel("number of queries", fontsize=fontsize)
            ax1.set_ylabel("objective value", fontsize=fontsize)
            ax1.legend(loc="lower right", fontsize=fontsize - 5)
            ax1.set_title(title, fontsize=fontsize)
            if in_sample:
                fig1.savefig(
                    problem_results_dir + problem_filename + "_is.pdf",
                    bbox_inches="tight",
                )
            else:
                fig1.savefig(
                    problem_results_dir + problem_filename + ".pdf", bbox_inches="tight"
                )
            # fig1.savefig(problem_results_dir + problem + ".pdf", bbox_inches="tight")
            # fig1.savefig(problem_results_dir + problem + ".eps", bbox_inches="tight")
