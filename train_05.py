import data_loader
import models
import torch
import torch.nn as nn
import util
from args import TrainArgParser
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver
import torch.nn.functional as F


#w
import pandas as pd
import numpy as np
import sys
import os


#w 训练函数
def train(args, table):
    #w 控制参数打印次数
    count=0
    much=5000
    much_w = 500

    #w 基本不用改
    if args.ckpt_path and not args.use_pretrained:
        model, ckpt_info=ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        args.start_epoch=ckpt_info['epoch']+1
        print('已加载训练参数')
    else:
        model_fn=models.__dict__[args.model]
        model=model_fn(**vars(args))
        if args.use_pretrained:
            model.load_pretrained(args.ckpt_path, args.gpu_ids)
        model=nn.DataParallel(model, args.gpu_ids)
    model=model.to(args.device)
    model.train()

    #w optimizer and scheduler，不用改
    if args.use_pretrained or args.fine_tune:
        parameters=model.module.fine_tuning_parameters(args.fine_tuning_boundary, args.fine_tuning_lr)
    else:
        parameters=model.parameters()
    optimizer=util.get_optimizer(parameters, args)
    lr_scheduler=util.get_scheduler(optimizer, args)

    if args.ckpt_path and not args.use_pretrained and not args.fine_tune:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)

    #w logger，evaluator，saver，不需要动
    cls_loss_fn=util.get_loss_fn(is_classification=True, dataset=args.dataset, size_average=False)
    data_loader_fn=data_loader.__dict__[args.data_loader]
    train_loader=data_loader_fn(args, phase='train', is_training=True)
    logger=TrainLogger(args, len(train_loader.dataset), train_loader.dataset.pixel_dict)
    eval_loaders=[data_loader_fn(args, phase='val', is_training=False)]
    evaluator=ModelEvaluator(args.do_classify, args.dataset, eval_loaders, logger,\
                               args.agg_method, args.num_visuals, args.max_eval, args.epochs_per_eval)
    saver=ModelSaver(args.save_dir, args.epochs_per_save, args.max_ckpts, args.best_ckpt_metric, args.maximize_metric)

    #w 后两行不需要动
    while not logger.is_finished_training():
        logger.start_epoch()

        #w 提取图像和标签信息，第二行开启logger
        for inputs,target_dict in train_loader:
            logger.start_iter()
            
            #w 处理table，不需要动
            ids=[int(item) for item in target_dict['study_num']]
            table_data=[]
            for i in range(len(target_dict['study_num'])):
                table_data.append(torch.tensor(np.array(table[table['idx']==ids[i]].iloc[:,4:]), dtype=torch.float32))
            table_data=torch.stack(table_data).squeeze(1)

            #w 如果是tSNE，设置成False
            #w 如果不是tSNE，设置成True
            with torch.set_grad_enabled(True):

                #w data
                img = inputs.to(args.device)
                tab = table_data.to(args.device)
                label = target_dict['is_abnormal'].to(args.device)

                #w forward
                if logger.epoch == 1:
                    loss_dynamic = model(img, tab, label, logger.epoch)
                    loss = loss_dynamic
                else:
                    final, loss_dynamic, w_list = model(img, tab, label, logger.epoch)
                    if loss_dynamic != -1:
                        loss_final = cls_loss_fn(final, label).mean()
                        loss = loss_dynamic + loss_final
                    else:
                        loss_final = cls_loss_fn(final, label).mean()
                        loss = loss_final

                #w check
                count += 1
                if count % much_w == 0 and logger.epoch > 1:
                    print()
                    print('loss_dynamic: ', loss_dynamic)
                    print('loss_final: ', loss_final)
                    print('w_i: ', w_list[0])
                    print('i_res_logit: ', w_list[1])
                    print('w_t: ', w_list[2])
                    print('t_res_logit: ', w_list[3])
                    print()

                if(count%much)==0:
                    for name,param in model.named_parameters():
                        print(name,param)
                    much=much*10

                #w 三步走
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #w 后边三行都不要动
                logger.log_iter(inputs,None,None,loss,optimizer)

            logger.end_iter()
            util.step_scheduler(lr_scheduler, global_step=logger.global_step)

        #w avg_loss和metrics.get(args.best_ckpt_metric,None)这两需要换有时候，其他不用动
        metrics,curves, eval_loss=evaluator.evaluate(model,args.device,logger.epoch)
        saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,\
                   metric_val= eval_loss) #w metrics.get(args.best_ckpt_metric,None)


        #w 这些不需要动
        logger.end_epoch(metrics, curves)
        util.step_scheduler(lr_scheduler, metrics, epoch=logger.epoch, best_ckpt_metric=args.best_ckpt_metric)


#w 程序入口，只需要动的地方是table路径
if __name__ =='__main__':
    #w table路径
    table_data=pd.read_csv('/mntcephfs/lab_data/wangcm/wangzhipeng/ehr/ehr1_nosub.csv')

    #w 这些不用动
    util.set_spawn_enabled()
    parser=TrainArgParser()
    args_=parser.parse_args()
    train(args_, table_data)



