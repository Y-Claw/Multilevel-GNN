import sys
import os
import os.path as osp
import numpy as np
import math
import random
import time
import logging
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, random_split, Subset
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DataParallel
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import statistics

from opt import parse_opts
from models import get_model
from dataloader.multiloader import MyData
from utils.ckpt_util import save_ckpt
from utils.cache_data import have_cached_data, cache_data, get_cached_data

def set_all_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False #
    cudnn.deterministic = True #
    torch.cuda.manual_seed(seed) #
    torch.cuda.manual_seed_all(seed)

#set_all_seed(1)

def train(model, device, loader, optimizer, criterion, criterion_weight, args):
    loss_list = []
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        torch.cuda.empty_cache()
        """if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:"""
        model.step = step
        pred, pca_feature = model(batch)
        loss_feature = model.get_feature_loss(pca_feature)
        optimizer.zero_grad()
        #if model.epoch == 8 and step == 1:
        #    import pdb; pdb.set_trace()
        if args.weighted_loss:
            loss_weight = criterion_weight[torch.arange(len(batch.y)//2),(batch.y.reshape(-1, 2)[:, 1]==1).to(int)][:, None].to(batch.y.device)
            loss = (loss_weight * criterion(pred.to(torch.float32), batch.y.reshape(-1, 2).to(torch.float32))).mean()
        elif args.batch_weighted_loss:
            loss_weight = criterion_weight[torch.arange(len(batch.y)//2),(batch.y.reshape(-1, 2)[:, 1]==1).to(int)][:, None].to(batch.y.device).mean()
            loss = (loss_weight * criterion(pred.to(torch.float32), batch.y.reshape(-1, 2).to(torch.float32)))
        else:
            loss = criterion(pred.to(torch.float32), batch.y.reshape(-1, 2).to(torch.float32))
        loss += loss_feature
        logging.info("loss train: {}".format(loss.item()))
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        loss_list.append(loss.item())
    return statistics.mean(loss_list)


@torch.no_grad()
def eval(model, device, loader, evaluator, criterion, args):
    model.eval()
    import pdb
    #pdb.set_trace()
    y_true = []
    y_pred = []
    y_pred_auc = []
    y_pred_acc = []
    loss_list = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            print('error')
            pass
        else:
            pred, pca_feature = model(batch)
            loss = criterion(pred.to(torch.float32), batch.y.reshape(-1, 2).to(torch.float32))
            loss_list.append(loss.item())
            y_true.append(batch.y.view(-1, 2).detach().cpu())
            #y_pred_auc.append(torch.softmax(pred.detach(), dim=-1)[:, 1].view(-1, 1).cpu())
            y_pred_auc.append(pred.detach().view(-1, 2).cpu())
            y_pred_acc.append(pred.detach()[:, 0].view(-1, 1).cpu() > 0.5)
            #y_pred.append(pred.detach().view(-1, 2).cpu())
            if args.metrics == 'auc':
                y_pred.append(pred.detach().view(-1, 2).cpu())
                #y_pred.append(torch.softmax(pred.detach(), dim=-1).view(-1, 1).cpu())
            elif args.metrics == 'acc':
                #y_pred.append(torch.argmax(pred.detach(), dim=-1).view(-1, 1).cpu())
                y_pred.append(pred.detach()[:, 0].view(-1, 1).cpu() > 0.5)

    y_true = torch.cat(y_true, dim=0).numpy()[:,0] >= 0.5
    y_pred = torch.cat(y_pred, dim=0).numpy()[:,0]
    y_pred_auc = torch.cat(y_pred_auc, dim=0).numpy()[:,0]
    y_pred_acc = torch.cat(y_pred_acc, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred_auc}

    return evaluator(y_true, y_pred), accuracy_score(y_true, y_pred_acc), roc_auc_score(y_true, y_pred_auc) , input_dict, statistics.mean(loss_list)

def run(model, device, train_loader, valid_loader, test_loader, criterion_weight, evaluator, check_epochs, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.wd)
    if args.step > 0:
        scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    if args.weight_balance:
        #logging.info("weight: {}".format(criterion_weight))
        #criterion_weight = torch.tensor([1, 16])
        criterion = torch.nn.BCELoss(weight=criterion_weight).to(device)
        #criterion = torch.nn.CrossEntropyLoss(weight=criterion_weight).to(device)
    elif args.weighted_loss:
        criterion = torch.nn.BCELoss(reduction='none')
    else:
        criterion = torch.nn.BCELoss()
        #criterion = torch.nn.CrossEntropyLoss()
    criterion_weightless = torch.nn.BCELoss()
    results = {'highest_valid': -1,
            'highest_valid_loss': 100,
            'final_train': -1,
            'final_test': -1,
            'highest_train': -1,
            'result_y':{},
            'epoch_result': {},
            'epoch_result_by_loss': {},
            'epoch_result_by_epoch': {}}

    """for check_epoch in check_epochs:
        results['epoch_result_by_loss'].setdefault(epoch, [])
        results['epoch_result'].setdefault(epoch, [])"""

    start_time = time.time()

    evaluate = True

    for epoch in range(1, args.epochs + 1):
        logging.info("=====Epoch {}".format(epoch))
        logging.info('Training...')
        model.epoch = epoch
        epoch_loss = train(model, device, train_loader, optimizer, criterion, criterion_weight, args)
        
        """if args.num_layers > args.num_layers_threshold:
            if epoch % args.eval_steps != 0:
                evaluate = False
            else:
                evaluate = True"""

        #model.print_params(epoch=epoch)

        if evaluate:

            logging.info('Evaluating...')
            torch.cuda.empty_cache()
            train_eval, train_accuracy, train_auc, train_res, train_loss = eval(model, device, train_loader, evaluator, criterion_weightless, args)
            torch.cuda.empty_cache()
            valid_eval, valid_accuracy, valid_auc, valid_res, valid_loss = eval(model, device, valid_loader, evaluator, criterion_weightless, args)
            torch.cuda.empty_cache()
            test_eval, test_accuracy, test_auc, test_res, test_loss = eval(model, device, test_loader, evaluator, criterion_weightless, args)
            torch.cuda.empty_cache()

            logging.info({'epoch_loss': epoch_loss,
                        'Train_acc': train_accuracy,
                        'Validation_acc': valid_accuracy,
                        'Test_acc': test_accuracy,
                        'Train_auc': train_auc,
                        'Validation_auc': valid_auc,
                        'Test_auc': test_auc,
                        'Valid_loss': valid_loss})

            if train_eval > results['highest_train']:

                results['highest_train'] = train_eval

            if valid_loss < results['highest_valid_loss']:
                results['highest_valid_loss'] = valid_loss
                results['final_test_by_loss'] = test_eval
                results['result_y_by_loss'] = test_res

            if valid_eval > results['highest_valid']:
                results['highest_valid'] = valid_eval
                results['final_train'] = train_eval
                results['final_test'] = test_eval
                results['result_y'] = test_res

                # save_ckpt(model, optimizer,
                #         round(epoch_loss, 4), epoch,
                #         args.model_save_path,
                #         args.name_pre, name_post='valid_best')

            if epoch in check_epochs:
                results['epoch_result_by_loss'][epoch] = results['result_y_by_loss']
                results['epoch_result'][epoch] = results['result_y']
                results['epoch_result_by_epoch'][epoch]  = test_res

            # if train_accuracy == 1:
            #     break
        if args.step > 0:
            scheduler.step()
    logging.info("%s" % results['final_test_by_loss'])
    logging.info("%s" % results['final_test'])

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))
    return results

def main():
    args = parse_opts()
    args.model_save_path = osp.join(args.model_save_path, args.save_dir, time.strftime('%Y-%m-%d-%H-%M-%S-', time.localtime(time.time())) + args.save_tag) if not args.debug else osp.join(args.model_save_path, 'debug')
    set_all_seed(args.seed)
    
    if not osp.exists(args.model_save_path):
        os.mkdir(args.model_save_path)
    
    logging.basicConfig(filename=osp.join(args.model_save_path, 'train.log'),level=logging.DEBUG)

    with open(osp.join(args.model_save_path, "command.txt"), 'w') as save:
        save.write("python " + " ".join(sys.argv))
        save.close()

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')
    dataset = MyData(args.raw_mrna_path.format(args.cancer_type), args.raw_cnv_path.format(args.cancer_type), args.raw_methylation_path.format(args.cancer_type),
                     args.node_path, args.edge_path.format(args.cancer_type), args.kegg_path, args.clinical_path.format(args.cancer_type), args)
    args.node_num = dataset.get_node_num()
    args.omics_num = len(dataset.omics_types)
    check_epochs = [30,35,40,45,50]
    check_epochs = list(range(5, args.epochs+1, 5))
    final_results = {}
    final_results_by_loss = {}
    final_results_by_epoch = {}
    final_acc_results = {}
    final_acc_results_by_loss = {}
    final_acc_results_by_epoch = {}
    for check_epoch in check_epochs:
        final_results.setdefault(check_epoch, [])
        final_results_by_loss.setdefault(check_epoch, [])
        final_results_by_epoch.setdefault(check_epoch, [])
        final_acc_results.setdefault(check_epoch, [])
        final_acc_results_by_loss.setdefault(check_epoch, [])
        final_acc_results_by_epoch.setdefault(check_epoch, [])
    for t in range(args.num_run):
        test_num = int(len(dataset) * 0.2)
        valid_num = int((len(dataset) - test_num) * 0.2)
        train_num = len(dataset) - test_num - valid_num
        #lgg_random_state = 
        if not args.split_shaffle:
            skf_tune_test = StratifiedKFold(n_splits=5, shuffle=args.split_shaffle)
            skf_train_valid = StratifiedKFold(n_splits=5, shuffle=args.split_shaffle)
        else:
            skf_tune_test = StratifiedKFold(n_splits=5, shuffle=args.split_shaffle,random_state=args.split_seed)
            skf_train_valid = StratifiedKFold(n_splits=5, shuffle=args.split_shaffle,random_state=args.split_seed)
        labels = dataset.get_labels()
        total_label = []
        total_pred = {}
        total_pred_by_loss = {}
        total_pred_by_epoch = {}
        for check_epoch in check_epochs:
            total_pred.setdefault(check_epoch, [])
            total_pred_by_loss.setdefault(check_epoch, [])
            total_pred_by_epoch.setdefault(check_epoch, [])
        all_idxs = np.array(range(len(labels)))
        #np.random.shuffle(shuffled_idx)
        count = 0
        for fold_i, (tune_data, test_idx) in enumerate(skf_tune_test.split(all_idxs,labels[all_idxs])):
            logging.info('total patient num {}'.format(len(all_idxs)))
            logging.info('label 0 number {}, label 1 number {}'.format(sum(labels==0), sum(labels==1)))
            train_idx, valid_idx = next(skf_train_valid.split(tune_data,labels[tune_data]))
            train_idx = tune_data[train_idx]
            valid_idx = tune_data[valid_idx]
            #args.num_tasks = dataset.num_classes
            if args.metrics == 'acc':
                evaluator = accuracy_score
            elif args.metrics == 'auc':
                evaluator = roc_auc_score

            logging.info('%s' % args)

            #split_idx = dataset.get_idx_split()
            model = get_model(args.model)(args)
            
            model.set_pathway_indexs(dataset.all_indice.to(device))
            x, y = dataset.get_data_by_indice(train_idx)
            tf_token = dataset.get_tf_token()
            mutual_info_mask, mutual_info = model.generate_mutual_mask(x, y, args.mutual_classif, fold_i, tf_token)
            model.set_info_mask(mutual_info_mask)
            dataset.recalculate_pca_bo_selected_gene(mutual_info_mask)
            model.set_pca_params(dataset.pca_components.to(device), mutual_info_mask)
            dataset.recalculate_edge_bo_selected_gene(mutual_info_mask, train_idx)
            if args.reorder_pathway:
                model.set_reorder_idxs(dataset.get_reorder_idxs())

            train_dataset = Subset(dataset, train_idx)
            valid_dataset = Subset(dataset, valid_idx)
            test_dataset = Subset(dataset, test_idx)

            criterion_weight = dataset.get_weight_balance(train_dataset.indices)

            if args.class_sample:
                _, y = dataset.get_data_by_indice(train_idx)
                weights = [criterion_weight[0][int(label > 0.5)] for label in y]
                #num_samples = len(train_idx)
                #num_samples = args.batch_size * ((len(train_idx)) // args.batch_size)
                num_samples = args.batch_size * math.ceil((len(train_idx)) / args.batch_size)
                sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False, sampler=sampler)
            elif args.weighted_loss or args.batch_weighted_loss:
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, drop_last=False)
            else:
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, drop_last=True)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

            model = model.to(device)

            #logging.info(model)
            result = run(model, device, train_loader, valid_loader, test_loader, criterion_weight, evaluator, check_epochs, args)
            total_label.append(result['result_y']["y_true"])
            for check_epoch in check_epochs:
                total_pred[check_epoch].append(result['epoch_result'][check_epoch]['y_pred'])
                total_pred_by_loss[check_epoch].append(result['epoch_result_by_loss'][check_epoch]['y_pred'])
                total_pred_by_epoch[check_epoch].append(result['epoch_result_by_epoch'][check_epoch]['y_pred'])
        label = np.concatenate(total_label)
        for check_epoch in check_epochs:
            epoch_pred = np.concatenate(total_pred[check_epoch])
            epoch_loss_pred = np.concatenate(total_pred_by_loss[check_epoch])
            epoch_epoch_pred = np.concatenate(total_pred_by_epoch[check_epoch])

            final_result = roc_auc_score(label,epoch_pred)
            final_results[check_epoch].append(final_result)

            final_result_by_loss = roc_auc_score(label,epoch_loss_pred)
            final_results_by_loss[check_epoch].append(final_result_by_loss)

            final_result_by_epoch = roc_auc_score(label,epoch_epoch_pred)
            final_results_by_epoch[check_epoch].append(final_result_by_epoch)

            final_acc_results[check_epoch].append(sum(label == (epoch_pred > 0.5)) / len(label))
            final_acc_results_by_loss[check_epoch].append(sum(label == (epoch_loss_pred > 0.5)) / len(label))
            final_acc_results_by_epoch[check_epoch].append(sum(label == (epoch_epoch_pred > 0.5)) / len(label))
            logging.info('experiment {} epoch {}: {} {} {} {} {} {}'.format(t, check_epoch, final_result, final_result_by_loss, final_result_by_epoch, final_acc_results[check_epoch], final_acc_results_by_loss[check_epoch], final_acc_results_by_epoch[check_epoch]))
        #torch.cuda.empty_cache()
    
    for i, final_result in enumerate(final_results[args.epochs]):
        logging.info('experiment {}: {}'.format(i, final_result))
    
    for i, final_result in enumerate(final_results_by_loss[args.epochs]):
        logging.info('experiment by loss {}: {}'.format(i, final_result))

    for i, final_result in enumerate(final_results_by_epoch[args.epochs]):
        logging.info('experiment by epoch {}: {}'.format(i, final_result))

    for check_epoch in check_epochs:
        logging.info('Avg AUC Score epoch {}: {}'.format(check_epoch, np.mean(final_results[check_epoch])))
        logging.info('Avg AUC Score by loss epoch {}: {}'.format(check_epoch, np.mean(final_results_by_loss[check_epoch])))
        logging.info('Avg AUC std by loss epoch {}: {}'.format(check_epoch, np.std(final_results_by_loss[check_epoch])))
        logging.info('Avg AUC Score by epoch epoch {}: {}'.format(check_epoch, np.mean(final_results_by_epoch[check_epoch])))
        logging.info('Avg AUC std by epoch epoch {}: {}'.format(check_epoch, np.std(final_results_by_epoch[check_epoch])))
        logging.info('Avg ACC Score epoch {}: {}'.format(check_epoch, np.mean(final_acc_results[check_epoch])))
        logging.info('Avg ACC Score by loss epoch {}: {}'.format(check_epoch, np.mean(final_acc_results_by_loss[check_epoch])))
        logging.info('Avg ACC Score by epoch epoch {}: {}'.format(check_epoch, np.mean(final_acc_results_by_epoch[check_epoch])))
    results = {
        "final_results": final_results,
        "final_results_by_loss": final_results_by_loss,
        "final_results_by_epoch": final_results_by_epoch,
    }
    torch.save(results, osp.join(args.model_save_path, "results.pth"))

if __name__ == '__main__':
    main()