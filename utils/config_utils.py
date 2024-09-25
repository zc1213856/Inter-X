import os
import yaml
import configargparse

import subprocess as sp

GPU_MINIMUM_MEMORY = 5500

def save_args(opt, folder):
    os.makedirs(folder, exist_ok=True)
    
    # Save as yaml
    optpath = os.path.join(folder, "opt.yaml")
    with open(optpath, 'w') as opt_file:
        yaml.dump(opt, opt_file)


def get_gpu_device():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    for gpu_idx, free_mem in enumerate(memory_free_values):
        if free_mem > GPU_MINIMUM_MEMORY:
            return gpu_idx
    Exception('No GPU with required memory')

def load_args(filename):
    with open(filename, "rb") as optfile:
        opt = yaml.load(optfile, Loader=yaml.Loader)
    return opt

def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'SEU-VCL GMR project'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='GMR',)

    parser.add_argument('--trainset',
                        default='',
                        type=str,
                        help='trainset.')
    parser.add_argument('--testset',
                        default='',
                        type=str,
                        help='testset.')
    parser.add_argument('--mae_dir',
                        default='',
                        type=str,
                        help='mae_dir.')
    parser.add_argument('--data_folder',
                        default='',
                        help='The directory that contains the data.')
    parser.add_argument('--keyp_folder',
                        default='',
                        help='The directory that contains the keypoints.')
    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--note',
                        default='test',
                        type=str,
                        help='code note')
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='learning rate.')
    parser.add_argument('--batchsize',
                        default=10,
                        type=int,
                        help='batch size.')
    parser.add_argument('--frame_length',
                        default=16,
                        type=int,
                        help='frame length.')
    parser.add_argument('--num_joint',
                        default=24,
                        type=int,
                        help='num_joint.')
    parser.add_argument('--epoch',
                        default=500,
                        type=int,
                        help='num epoch.')
    parser.add_argument('--worker',
                        default=0,
                        type=int,
                        help='workers for dataloader.')
    parser.add_argument('--mode',
                        default='',
                        type=str,
                        help='running mode.')        
    parser.add_argument('--pretrain',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for use pretrain parameters.')
    parser.add_argument('--use_prior',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for use pretrain parameters.')
    parser.add_argument('--pretrain_dir',
                        default='',
                        type=str,
                        help='The directory that contains the pretrain model.')
    parser.add_argument('--model_dir',
                        default='',
                        type=str,
                        help='(if test only) The directory that contains the model.')
    parser.add_argument('--model',
                        default='',
                        type=str,
                        help='the model used for this project.')
    parser.add_argument('--train_loss',
                        default='L1 partloss',
                        type=str,
                        help='training loss type.')
    parser.add_argument('--test_loss',
                        default='L1',
                        type=str,
                        help='testing loss type.')
    parser.add_argument('--viz',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for visualize input.')
    parser.add_argument('--task',
                        default='ed_train',
                        type=str,
                        help='ee_train: encoder-encoder only, else ed_train.')
    parser.add_argument('--gpu_index',
                        default=0,
                        type=int,
                        help='gpu index.')
    parser.add_argument('--num_neurons',
                        default=512,
                        type=int,
                        help='num_neurons.')
    parser.add_argument('--latentD',
                        default=32,
                        type=int,
                        help='latentD.')
    parser.add_argument('--data_shape',
                        default=21,
                        type=int,
                        help='data_shape.')
    parser.add_argument('--kl_coef',
                        default=5e-3,
                        type=float,
                        help='kl_coef.')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict
