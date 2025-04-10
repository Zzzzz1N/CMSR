import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer.trainer import PretrainTrainer
from recbole.utils import init_seed, init_logger
from model.cmsr import CMSR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=1.2, type=float, help='alpha in WCE')
    parser.add_argument('--beta', default=0.99, type=float)
    parser.add_argument('--loss_type', default='WCE', type=str, help='loss type: CE, WCE')
    parser.add_argument('--dataset', default='us', type=str, help='the source market to pretrain, default is us')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--pretrain_epochs', default=200, type=int, help='pretrain epochs')
    parser.add_argument('--with_adapter', default=False, type=bool, help='whether to use adapter')
    parser.add_argument('--checkpoint_dir', default='saved/', type=str, help='checkpoint dir')
    parser.add_argument('--save_step', default=50, type=int, help='save step')
    args = parser.parse_args()
    
    
    config_dict = {}
    config_dict['dataset'] = args.dataset
    config_dict['alpha'] = args.alpha # 1.2 1.4 1.6
    config_dict['beta'] = args.beta # 0.5 1.0 1.5
    config_dict['loss_type'] = args.loss_type
    config_dict['gpu_id'] = args.gpu_id
    config_dict['save_step'] = args.save_step
    config_dict['pretrain_epochs'] = args.pretrain_epochs
    config_dict['checkpoint_dir'] = args.checkpoint_dir
    config_dict['with_adapter'] = False
    
    # configurations initialization
    config = Config(model=CMSR, config_dict=config_dict, config_file_list=['config/CMSR.yaml', 'config/DATA.yaml'])

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = CMSR(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # # Check total parameters and trainable parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    
    # trainer loading and initialization
    trainer = PretrainTrainer(config, model)
    trainer.pretrain(train_data, show_progress=False)
    print("Pretrain Done!")