import data_loader
import models
import torch
import torch.nn as nn
import util
from torch.autograd import Variable

from args import TrainArgParser
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver

###
import pandas as pd
import numpy as np
import sys

###
def train(args,table):
    ###
    count=0
    much=1000

    if args.ckpt_path and not args.use_pretrained:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        ###
        model = model_fn(**vars(args))
        if args.use_pretrained:
            model.load_pretrained(args.ckpt_path, args.gpu_ids)
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Get optimizer and scheduler
    if args.use_pretrained or args.fine_tune:
        parameters = model.module.fine_tuning_parameters(args.fine_tuning_boundary, args.fine_tuning_lr)
    else:
        parameters = model.parameters()
    optimizer = util.get_optimizer(parameters, args)
    lr_scheduler = util.get_scheduler(optimizer, args)

    #W Discriminator
    D_fn=models.__dict__['D']
    D=D_fn(**vars(args))
    D=nn.DataParallel(D,args.gpu_ids)
    D=D.to(args.device)
    D.train()
    D_parameters=D.parameters()
    D_optimizer=util.get_optimizer(D_parameters,args)
    D_lr_scheduler=util.get_scheduler(D_optimizer,args)

    if args.ckpt_path and not args.use_pretrained and not args.fine_tune:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, D_lr_scheduler)

    # Get logger, evaluator, saver
    cls_loss_fn = util.get_loss_fn(is_classification=True, dataset=args.dataset, size_average=False)
    data_loader_fn = data_loader.__dict__[args.data_loader]
    train_loader = data_loader_fn(args, phase='train', is_training=True)
    logger = TrainLogger(args, len(train_loader.dataset), train_loader.dataset.pixel_dict)
    eval_loaders = [data_loader_fn(args, phase='val', is_training=False)]
    evaluator = ModelEvaluator(args.do_classify, args.dataset, eval_loaders, logger,
                               args.agg_method, args.num_visuals, args.max_eval, args.epochs_per_eval)
    saver = ModelSaver(args.save_dir, args.epochs_per_save, args.max_ckpts, args.best_ckpt_metric, args.maximize_metric)

    #W
    D_loss=nn.BCELoss()

    # Train model
    while not logger.is_finished_training():
        logger.start_epoch()

        for inputs, target_dict in train_loader:
            logger.start_iter()
            
            ###
            ids = [int(item) for item in target_dict['study_num']]
            table_data=[]
            for i in range(len(target_dict['study_num'])):
                table_data.append(torch.tensor(np.array(table[table['idx']==ids[i]].iloc[:,4:]),dtype=torch.float32))
            table_data = torch.stack(table_data).squeeze(1)
 
            with torch.set_grad_enabled(True):
                #W Data
                inputs=inputs.to(args.device)
                table_data=table_data.to(args.device)
                cls_targets=target_dict['is_abnormal'].to(args.device)
                
                #W Model for logit
                #W cls_logits=model.forward(inputs)
                #W cls_loss=cls_loss_fn(cls_logits,cls_targets)
                #W loss=cls_loss.mean()

                #W Model for Final
                optimizer.zero_grad()

                valid=Variable(torch.cuda.FloatTensor(args.batch_size,1).fill_(1.0),requires_grad=False)
                fake=Variable(torch.cuda.FloatTensor(args.batch_size,1).fill_(0.0),requires_grad=False)
                F_loss,i_out,t_out,loss_clip,p_out,d_i_sh,d_t_sh=model(inputs,table_data,cls_targets)

                g_loss=D_loss(D(d_i_sh),valid)
                p_loss=cls_loss_fn(p_out,cls_targets).mean()
                out_loss=cls_loss_fn(i_out,cls_targets).mean()+cls_loss_fn(t_out,cls_targets).mean()
                loss=(F_loss+loss_clip+p_loss+out_loss+g_loss)/4

                loss.backward()
                optimizer.step()

                #W
                D_optimizer.zero_grad()

                real_loss=D_loss(D(d_t_sh.detach()),valid)
                fake_loss=D_loss(D(d_i_sh.detach()),fake)
                d_loss=(real_loss+fake_loss)/2

                d_loss.backward()
                D_optimizer.step()

                #W
                count+=1
                #W 检查loss
                #W if(count%1000)==0:
                #W     print('clip_loss:',loss_inter.item(),' ',loss_global.item(),' ',loss_remedy_global.item())
                #W 检查参数有没有动
                if(count%much)==0:
                    for name,param in model.named_parameters():
                        print(name,param)
                    much=much*5
        
                ###Log
                logger.log_iter(inputs,None,None,loss,optimizer)

            logger.end_iter()
            util.step_scheduler(lr_scheduler,global_step=logger.global_step)
            util.step_scheduler(D_lr_scheduler,global_step=logger.global_step)

        ###
        metrics,curves,avg_loss=evaluator.evaluate(model,args.device,logger.epoch)
        saver.save(logger.epoch,model,optimizer,lr_scheduler,args.device,
                    metric_val=metrics.get(args.best_ckpt_metric,None))#W metrics.get(args.best_ckpt_metric,None)

        logger.end_epoch(metrics,curves)
        util.step_scheduler(lr_scheduler, metrics, epoch=logger.epoch, best_ckpt_metric=args.best_ckpt_metric)
        util.step_scheduler(D_lr_scheduler, metrics, epoch=logger.epoch, best_ckpt_metric=args.best_ckpt_metric)

###
if __name__ == '__main__':
    ###
    table_data = pd.read_csv('/mntcephfs/lab_data/wangcm/wangzhipeng/ehr/ehr1.csv')

    util.set_spawn_enabled()
    parser = TrainArgParser()
    args_ = parser.parse_args()
    ###
    train(args_,table_data)
