'''Main pretraining script.'''

import os
import shutil

import flatten_dict
import hydra

from src.datasets.catalog import PRETRAINING_DATASETS, UNLABELED_DATASETS

from clearml import Task

def run_pytorch(config):
    '''Runs pretraining in PyTorch.'''
    import pytorch_lightning as pl

    from src.evaluators.pytorch import online_evaluator, pretraining_output_visualiser
    from src.systems.pytorch import contpred, emix, mae, shed

    # Check for dataset.
    assert config.dataset.name in PRETRAINING_DATASETS, f'{config.dataset.name} not one of {PRETRAINING_DATASETS}.'

    # Set up config, callbacks, loggers.
    flat_config = flatten_dict.flatten(config, reducer='dot')
    save_dir = os.path.join(config.exp.base_dir, config.exp.name)

    # Set RNG.
    pl.seed_everything(config.trainer.seed)
    
    # logger
    tb_logger = pl.loggers.TensorBoardLogger(save_dir)
    tb_logger.log_hyperparams(flat_config)
    callbacks = [
        pl.callbacks.ModelCheckpoint(dirpath=save_dir, every_n_epochs=config.trainer.ckpt_every_n_epochs, save_top_k=-1)
    ]

    # Initialize training module.
    if config.algorithm == 'emix':
        system = emix.EMixSystem(config)
    elif config.algorithm == 'shed':
        system = shed.ShEDSystem(config)
    elif config.algorithm == 'capri':
        system = contpred.ContpredSystem(config, negatives='sequence')
    elif config.algorithm == 'mae':
        system = mae.MAESystem(config)
    else:
        raise ValueError(f'Unimplemented algorithm config.algorithm={config.algorithm}.')

    # Online evaluator for labeled datasets.
    if config.dataset.name not in UNLABELED_DATASETS:
        ssl_online_evaluator = online_evaluator.SSLOnlineEvaluator(
            dataset=config.dataset.name,
            metric=config.dataset.metric,
            loss=config.dataset.loss,
            z_dim=config.model.kwargs.dim,
            num_classes=system.dataset.num_classes()
        )
        callbacks += [ssl_online_evaluator]
        
    callbacks += [pretraining_output_visualiser.PretrainingOutputVisualiser(config)]
    callbacks += [pl.callbacks.LearningRateMonitor(logging_interval='step')]

    # PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=tb_logger,
        accelerator='gpu',
        devices=config.gpus,
        max_epochs=config.trainer.max_epochs,
        # min_epochs=config.trainer.min_epochs,
        # resume_from_checkpoint=config.trainer.resume_from_checkpoint,
        val_check_interval=config.trainer.val_check_interval,
        limit_val_batches=config.trainer.limit_val_batches,
        callbacks=callbacks,
        # weights_summary=config.trainer.weights_summary,
        gradient_clip_val=config.trainer.gradient_clip_val,
        precision=config.trainer.precision,
        # overfit_batches=10,
        strategy="ddp",
        # track_grad_norm=2,
        replace_sampler_ddp=False,
    )

    trainer.fit(system)


def print_pytorch_info():
    import torch
    import torchaudio
    import torchvision
    header = '==== Using Framework: PYTORCH ===='
    print(header)
    print(f'   - [torch]       {torch.__version__}')
    print(f'   - [torchvision] {torchvision.__version__}')
    print(f'   - [torchaudio]  {torchaudio.__version__}')
    print('=' * len(header))


@hydra.main(config_path='conf', config_name='pretrain', version_base='1.1')
def run(config):
    '''Wrapper around actual run functions to import and run for specified framework.'''
    # task = Task.init(project_name='mae-qxr', task_name=config.exp.name)
    # logger = task.get_logger(save_dir)
    if config.framework == 'pytorch':
        print_pytorch_info()
         # write code to save the whole codebase of this directory to the condig.exp.base_dir
    # save the codebase to the clearml server
        current_directory = os.getcwd()
        destination_directory = os.path.join(config.exp.base_dir, config.exp.name)
        codebase_directory = os.path.join(destination_directory, 'codebase')
        print(codebase_directory)
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
        # os.makedirs(codebase_directory)
        # if os.path.exists(codebase_directory):
        #     print("The directory 'codebase' already exists in the base directory. Please delete or rename it before running this script again.")
        #     exit(0)
        # else:
        #     # copying directory trees - this will copy all files and subdirectories, and will overwrite any conflicts
        #     shutil.copytree(current_directory, codebase_directory)
        #     print(f"Successfully copied all files and directories from {current_directory} to {codebase_directory}.")
        run_pytorch(config)
    else:
        raise ValueError(f'Framework {config.framework} not supported.')


if __name__ == '__main__':
    run()
