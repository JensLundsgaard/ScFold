import logging
import pickle
import json
import torch
import os.path as osp
import numpy as np

import warnings

from tqdm import tqdm

from methods.utils import cuda

warnings.filterwarnings('ignore')

from methods import ProDesign
from API import Recorder
from utils import *

#print(torch.cuda.is_available())
class Exp:
    def __init__(self, args, show_params=True):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        if show_params:
            print_log(output_namespace(self.args))
    
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU:',device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    #以下负责设置实验的基本配置、创建保存结果的目录和日志文件、保存模型参数配置，并准备数据和构建方法
    def _preparation(self):
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        
        self._get_data()
        self._build_method()

    def _build_method(self):
        steps_per_epoch = 1000
        if self.args.method == 'ProDesign':
            self.method = ProDesign(self.args, self.device, steps_per_epoch)

    def _get_data(self):
        self.train_loader, self.valid_loader, self.test_loader = get_dataset(self.config)
    
    def train(self):
        recorder = Recorder(self.args.patience, verbose=True)
        for epoch in range(self.args.epoch):
            train_loss, train_perplexity = self.method.train_one_epoch(self.train_loader)

            if epoch % self.args.log_step == 0:
                with torch.no_grad():
                    valid_losses, valid_acc_1s, valid_acc_5s, valid_acc_10s = self.valid(epoch)

                    # self._save(name=str(epoch))
                    #self.test()
                
                print(f" {sum(params.numel() for params in self.method.model.parameters() if params.requires_grad)} & ${valid_acc_1s.mean():.3f} \\pm {valid_acc_1s.std():.3f}$ & ${valid_acc_5s.mean():.3f} \\pm {valid_acc_5s.std():.3f}$ & ${valid_acc_10s.mean():.3f} \\pm {valid_acc_10s.std():.3f}$ & ${np.exp(valid_losses).mean():.3f} \\pm {np.exp(valid_losses).std():.3f}$ ")

                #recorder(valid_loss, self.method.model, self.path)
                #if recorder.early_stop:
                #    print("Early stopping")
                #    logging.info("Early stopping")
                #    break
            

        #best_model_path = osp.join(self.path, 'checkpoint.pth')
        #self.method.model.load_state_dict(torch.load(best_model_path))
        
    def valid(self, epoch):
        return self.method.valid_one_epoch(self.valid_loader, epoch=epoch)

    def test(self):
        test_perplexity, test_recovery, test_subcat_recovery = self.method.test_one_epoch(self.test_loader)
        print_log('Test Perp: {0:.4f}, Test Rec: {1:.4f}\n'.format(test_perplexity, test_recovery))

        for cat in test_subcat_recovery.keys():
            print_log('Category {0} Rec: {1:.4f}\n'.format(cat, test_subcat_recovery[cat]))

        return test_perplexity, test_recovery


if __name__ == '__main__':
    from parserdemo import create_parser
    args = create_parser()
    config = args.__dict__
    path = '/paper/demo/Zndesign-main/results/debug/zzw_38.pth'
    print(os.path.exists(path))  # 查看文件是否存在
    print(config)
    exp = Exp(args)
    
    #svpath = '/paper/demo/Zndesign-main/results/debug/'
    #exp.method.model.load_state_dict(torch.load(svpath+'zzw_38.pth'))

    print('>>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train()

    print('>>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    valid_perp, valid_rec = exp.valid()
