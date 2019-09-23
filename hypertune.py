import argparse
import logging
import os
import socket

from train import (
    get_train_parser,
    train,
)

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.core.worker import Worker
from hpbandster.optimizers import HyperBand


class PyTorchWorker(Worker):

    def __init__(self, args_train, **kwargs):
        super().__init__(**kwargs)
        self.args_train = args_train

    def compute(self, config, budget, *args, **kwargs):
        """ The input parameter "config" (dictionary) contains the sampled
        configurations passed by the bohb optimizer. """
        config["trainer_run:max_epochs"] = budget
        config["ignite_random:epoch_fraction"] = 0.1
        try:
            return train(self.args_train, config)
        except Exception as e:
            return {
                "loss": 1.0,
                "info": str(e),
            }

    @staticmethod
    def get_configspace():
        """ It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle
        categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.UniformFloatHyperparameter(
                'optimizer:lr',
                lower=0.001,
                upper=0.1,
                default_value=0.04,
                log=True,
            ),
            # CSH.OrdinalHyperparameter(
            #     'ignite_random:minibatch_size',
            #     sequence=[2, 4, 8, 16, 32],
            #     default_value=8,
            # ),
            # CSH.OrdinalHyperparameter(
            #     'ignite_random:num_minibatches',
            #     sequence=[2, 4, 8, 16, 32],
            #     default_value=8,
            # ),
            CSH.UniformIntegerHyperparameter(
                'model:history',
                lower=1,
                upper=12,
                default_value=12,
            ),
            CSH.UniformIntegerHyperparameter(
                'model:n_layers',
                lower=2,
                upper=8,
                default_value=3,
            ),
            CSH.OrdinalHyperparameter(
                'model:n_channels',
                sequence=[2, 4, 8, 16, 32, 64],
                default_value=8,
            ),
        ])
        return cs


def main():
    parser = argparse.ArgumentParser(
        parents=[get_train_parser()],
        description='Parallel execution of hyper-tuning',
    )
    parser.add_argument('--run-id',
                        required=True,
                        help='Name of the run')
    parser.add_argument('--min-budget',
                        type=float,
                        help='Minimum budget used during the optimization',
                        default=2)
    parser.add_argument('--max-budget',
                        type=float,
                        help='Maximum budget used during the optimization',
                        default=32)
    parser.add_argument('--n-iterations',
                        type=int,
                        help='Number of iterations performed by the optimizer',
                        default=1)
    parser.add_argument('--n-workers',
                        type=int,
                        help='Number of workers to run in parallel',
                        default=3)
    parser.add_argument('--eta',
                        type=int,
                        help='Parameter of the hyper-tuning algorithm',
                        default=3)
    parser.add_argument('--worker',
                        help='Flag to turn this into a worker process',
                        action='store_true')
    parser.add_argument('--hostname',
                        default=None,
                        help='IP of name server.')
    parser.add_argument('--shared-directory',
                        type=str,
                        help=('A directory that is accessible '
                              'for all processes, e.g. a NFS share'),
                        default='output/hypertune')
    args = parser.parse_args()
    print(args)

    if not args.hostname and socket.gethostname().lower().startswith('lenovo'):
        # If we are on cluster set IP
        args.hostname = hpns.nic_name_to_host('eno1')
    elif not args.hostname:
        args.hostname = '127.0.0.1'

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                        format='%(asctime)s %(message)s',
                        datefmt='%I:%M:%S')

    args.callbacks = ['learning-rate-scheduler', 'early-stopping']

    if args.worker:
        # Start a worker in listening mode (waiting for jobs from master)
        w = PyTorchWorker(
             args,
             run_id=args.run_id,
             host=args.hostname,
        )
        w.load_nameserver_credentials(working_directory=args.shared_directory)
        w.run(background=False)
        exit(0)

    result_logger = hpres.json_result_logger(
        directory=args.shared_directory,
        overwrite=True,
    )

    # Start a name server
    name_server = hpns.NameServer(
        run_id=args.run_id,
        host=args.hostname,
        port=0,
        working_directory=args.shared_directory,
    )
    ns_host, ns_port = name_server.start()

    # Run and optimizer
    bohb = HyperBand(
        configspace=PyTorchWorker.get_configspace(),  # model can be an arg here?
        run_id=args.run_id,
        result_logger=result_logger,
        eta=args.eta,
        host=args.hostname,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
    )

    res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    name_server.shutdown()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    inc_runs = res.get_runs_by_id(incumbent)
    inc_run = inc_runs[-1]
    all_runs = res.get_all_runs()

    print("Best loss {:6.2f}".format(inc_run.loss))
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
