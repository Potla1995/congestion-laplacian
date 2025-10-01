from experiments import *


def __main__():
    # Random Regular Experiments
    run_random_regular_experiment([3], list(range(10, 31)), 100, write_to_file=True)
    run_random_regular_experiment([4], list(range(20, 31)), 100, write_to_file=True)

    # Erdos-Renyi Experiments
    run_erdos_renyi_experiment([0.15],list(range(10, 25)), 50, write_to_file=True)
    run_erdos_renyi_experiment([0.2],list(range(10, 25)), 50, write_to_file=True)
    prob_list = [0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185]
    run_erdos_renyi_experiment(prob_list, [15], 50, write_to_file=True)

    # Random Circuits Experiments
    seeds_list = [0,1,2,3,5,7,9,11,13,15,17,19,20,21,22,23,24,25,26,27,28,29]+list(range(30,58))
    # Some seeds don't terminate, we basically reject the seeds that make the code run forever
    print(f'{len(seeds_list)} runs of RQC')
    run_random_qc_experiment([5], list(range(10, 21)), {1: 0, 2: 1, 3: 0, 4: 0}, seeds_list, write_to_file=True)
    run_random_qc_experiment([5], list(range(10, 21)), {1: 0, 2: 0, 3: 1, 4: 0}, seeds_list, write_to_file=True)
    run_random_qc_experiment(list(range(10, 21)), [10], {1: 0, 2: 1, 3: 0, 4: 0}, seeds_list, write_to_file=True)
    run_random_qc_experiment(list(range(10, 21)), [10], {1: 0, 2: 0, 3: 1, 4: 0}, seeds_list, write_to_file=True)

    print('All experiments done.')

__main__()

