import argparse
import logging
import os
import socket

from train import train

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.core.worker import Worker
from hpbandster.optimizers import HyperBand


class PyTorchWorker(Worker):

    def __init__(self, city, model_type, **kwargs):
        super().__init__(**kwargs)
        self.city = city
        self.model_type = model_type

    def compute(self, config, budget, *args, **kwargs):
        """ The input parameter "config" (dictionary) contains the sampled
        configurations passed by the bohb optimizer. """
        return train(self.city, self.model_type, config, max_epochs=budget)

    @staticmethod
    def get_configspace():
        """ It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle
        categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()
        lr = CSH.UniformFloatHyperparameter('optimizer_lr',
                                            lower=1e-6,
                                            upper=1e-1,
                                            default_value='1e-2',
                                            log=True)

        cs.add_hyperparameters([lr])
        return cs


def main():
    parser = argparse.ArgumentParser(
        description='Parallel execution of hyper-tuning')
    parser.add_argument('--run-id',
                        required=True,
                        help='Name of the run')
    parser.add_argument('--min-budget',
                        type=float,
                        help='Minimum budget used during the optimization',
                        default=1)
    parser.add_argument('--max-budget',
                        type=float,
                        help='Maximum budget used during the optimization',
                        default=10)
    parser.add_argument('--n-iterations',
                        type=int,
                        help='Number of iterations performed by the optimizer',
                        default=1)
    parser.add_argument('--n-workers',
                        type=int,
                        help='Number of workers to run in parallel',
                        default=3)
    parser.add_argument('--worker',
                        help='Flag to turn this into a worker process',
                        action='store_true')
    parser.add_argument('--hostname',
                        default='127.0.0.1', 
                        help='IP of name server.')
    parser.add_argument('--shared-directory',
                        type=str,
                        help=('A directory that is accessible '
                              'for all processes, e.g. a NFS share'),
                        default='output/hypertune')
    args = parser.parse_args()
    print(args)

    if socket.gethostname().startswith('lenovo'):
        args.hostname = '10.90.100.16'
        print(f"WARN Running on lenovo")
        print(f"WARN Changes hostname to {args.hostname}")

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                        format='%(asctime)s %(message)s',
                        datefmt='%I:%M:%S')

    if args.worker:
        # Start a worker in listening mode (waiting for jobs from master)
        w = PyTorchWorker("Berlin",
                          "temporal-regression-speed-12",
                          run_id=args.run_id,
                          id=0,
                          nameserver=args.hostname)
        w.run(background=False)
        exit(0)

    result_logger = hpres.json_result_logger(directory=args.shared_directory,
                                             overwrite=True)

    # Start a nameserver
    NS = hpns.NameServer(run_id=args.run_id, host=args.hostname, port=None)
    NS.start()

    # Run and optimizer
    bohb = HyperBand(
        configspace=PyTorchWorker.get_configspace(
        ),  # model can be an arg here?
        run_id=args.run_id,
        result_logger=result_logger,
        eta=3,
        min_budget=args.min_budget,
        max_budget=args.max_budget)

    res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    inc_runs = res.get_runs_by_id(incumbent)
    inc_run = inc_runs[-1]
    all_runs = res.get_all_runs()

    print("\nBEST loss {:6.2f}".format(1 - inc_run.loss))
    print('A total of %i unique configurations where sampled.' %
          len(id2config.keys()))
    print('A total of %i runs where executed.' % len(all_runs))
    print('Total budget corresponds to %.1f full function evaluations.' %
          (sum([r.budget for r in all_runs]) / args.max_budget))
    print('The run took  %.1f seconds to complete.' %
          (all_runs[-1].time_stamps['finished'] -
           all_runs[0].time_stamps['started']))

    # cs = worker.get_configspace()
    # config = cs.sample_configuration().get_dictionary()
    # print(config)
    # res = worker.compute(config=config, budget=1)
    # print(res)


if __name__ == "__main__":
    main()
