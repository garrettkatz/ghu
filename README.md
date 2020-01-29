Code used to generate results and figures in the paper:

"Reinforcement-based Program Induction in a Neural Virtual Machine"

We ran these experiments on a workstation with an 8-core Intel i7 CPU, 32GB of RAM, Fedora 30, Python 3.7.4, numpy 1.16.2, matplotlib 3.0.3, and pytorch 1.1.0.  With this configuration, the experiments took a matter of minutes or at most hours, with the longest experiments running and finishing overnight.  In a different configuration it may take longer or run out of memory.

The simplest script you can try is swap_experiments.py.  This will run 30 trials of the swap task, with very small memory footprint, in a few seconds.

The paper figures were generated by the following scripts.  These scripts also rerun the experiments and regenerate the results data, but you can also use the existing data in this archive and just plot.  If you want to do both, regenerate and plot:

swap_experiments.py -> Fig 1
echo_experiments.py -> Figs 2,3
max.py, filter.py, rfilter.py, then plot_filters.py -> Fig 4
recall_plastic.py -> Fig 5
reverse_with_repeats.py, then plot_reverse_dvc.py -> Figs 6-8

If you only want to plot, using the included data, you can run the foregoing, but first:

- For swap, echo, recall, and reverse: comment out the 3 lines right after the comment that says "Run the experiment"
- Skip max.py, filter.py, and rfilter.py (just run plot_filters.py)

