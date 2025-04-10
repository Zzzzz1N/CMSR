import argparse
import torch
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer.trainer import Trainer
from recbole.utils import init_seed, init_logger
from model.cmsr import CMSR


def load_param(path, weight_list):
    checkpoint = torch.load(path)
    print("Keys in checkpoint['state_dict']:")
    for param_tensor in checkpoint['state_dict']:
        print(param_tensor, "\t", checkpoint['state_dict'][param_tensor].size())
    weight_dict = dict()
    try:
        q_name = 'cmsr_encoder.layers.0.attention.w_q.weight'
        k_name = 'cmsr_encoder.layers.0.attention.w_k.weight'
        v_name = 'cmsr_encoder.layers.0.attention.w_v.weight'
        for weight in weight_list:
            if weight == 'query':
                weight_dict[weight] = checkpoint['state_dict'][q_name].detach().cpu()
            if weight == 'key':
                weight_dict[weight] = checkpoint['state_dict'][k_name].detach().cpu()
            if weight == 'value':
                weight_dict[weight] = checkpoint['state_dict'][v_name].detach().cpu()
    except KeyError:
        print("Raise KeyError: The state_dict does not contain the expected files.")
        key_list = list(checkpoint['state_dict'].keys())
        max_layer_num = 0
        for k in key_list:
            res = k.split('cmsr_encoder.layers.')
            if len(res) > 1:
                if int(res[1][0]) > max_layer_num:
                    max_layer_num = int(res[1][0])
        for weight in weight_list:
            name = 'cmsr_encoder.layers.{}.attention.{}.weight'.format(max_layer_num, weight)
            weight_dict[weight] = checkpoint['state_dict'][name].detach().cpu()
    return weight_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=1.2, type=float, help='alpha in WCE')
    parser.add_argument('--beta', default=0.99, type=float, help='beta in WCE')
    parser.add_argument('--loss_type', default='CE', type=str, help='loss type: CE, WCE')
    parser.add_argument('--dataset', default='ca', type=str, help='the source market to pretrain, default is us')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--epochs', default=200, type=int, help='pretrain epochs')
    parser.add_argument('--checkpoint_dir', default='saved/', type=str, help='checkpoint dir')
    parser.add_argument('--with_adapter', default=False, type=bool, help='whether to use adapter')
    parser.add_argument('--weight_path', default='saved/CATSR-us-200.pth', type=str, help='saved weight path')
    args = parser.parse_args()
    
    
    config_dict = {}
    config_dict['alpha'] = args.alpha # 1.2 1.4 1.6
    config_dict['beta'] = args.beta # 0.5 1.0 1.5
    config_dict['loss_type'] = args.loss_type
    config_dict['dataset'] = args.dataset
    config_dict['gpu_id'] = args.gpu_id
    config_dict['epochs'] = args.epochs
    config_dict['checkpoint_dir'] = args.checkpoint_dir
    config_dict['with_adapter'] = True
    
    # configurations initialization
    config = Config(model=CMSR, config_dict=config_dict, config_file_list=['config/CMSR.yaml', 'config/DATA.yaml'])

    weight_list = ["query", "key"]
    weight_path = args.weight_path
    weight_dict = load_param(weight_path, weight_list)
    
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

    model = CMSR(config, train_data.dataset, weight_dict).to(config['device'])
    logger.info(model)
    
    # trainer loading and initialization
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=False)
    
    # model evaluation
    test_result = trainer.evaluate(test_data)
    res_str = str()
    for k, v in test_result.items():
        res_str += str(v) + ' '
    print('CSV_easy_copy_format:\n', res_str)