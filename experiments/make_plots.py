import numpy as np
import os
import seaborn as sns
import sys

from matplotlib import pyplot as plt


problem = "ackley"

# problems
if problem == "dropwave":
    title = "dropwave"
    opt_val = 1.0
    n_iter = 50
elif problem == "ackley":
    title = "ackley"
    opt_val = 0.0
    n_iter = 20

n_trials = 30 # number of trials
problem_filename = problem
algos = ["qEUBO_Standard", "qEUBO_Composite"]
labels = ["Standard", "Composite"]
colors = ["green", "red"]
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
problem_results_dir = script_dir + "/results/" + problem + "/"


# matplotlib and seaborn settings
sns.set()
sns.set_palette("bright")
fontsize = 20
plt.rc("xtick", labelsize=fontsize)
plt.rc("ytick", labelsize=fontsize)

# this is used to plot all combinations between in-sample vs. out-of-sample regret and utility value vs. log_10 regret
in_sample_flags = [True, False]
log_regret_flags = [True, False]

for in_sample in in_sample_flags:
    for log_regret in log_regret_flags:
        fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        for a, algo in enumerate(algos):
            print(algo)
            n_iter_algo = n_iter[a] if isinstance(n_iter, list) else n_iter
            algo_results_dir = problem_results_dir + algo + "/"

            best_utility_vals_all_trials = np.zeros((n_trials, n_iter_algo + 1))

            for trial in range(1, n_trials + 1):
                print(trial)
                if in_sample:
                    file_tag = "max_utility_vals_within_queries_"
                else:
                    file_tag = "utility_vals_at_max_post_mean_"
                best_utility_vals_all_trials[trial - 1, :] = np.loadtxt(
                    algo_results_dir + file_tag + str(trial) + ".txt"
                )[: n_iter_algo + 1]

            print(np.max(best_utility_vals_all_trials))
            if log_regret:
                best_utility_vals_all_trials = np.log10(
                    opt_val - best_utility_vals_all_trials
                )

            mean = best_utility_vals_all_trials.mean(axis=0)
            sem = best_utility_vals_all_trials.std(axis=0) / np.sqrt(n_trials)

            ax1.plot(range(0, n_iter_algo + 1), mean, label=labels[a], color=colors[a])
            ax1.fill_between(
                range(0, n_iter_algo + 1),
                mean - 1.96 * sem,
                mean + 1.96 * sem,
                alpha=0.15,
                color=colors[a],
            )

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
