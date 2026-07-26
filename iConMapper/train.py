import os
import torch
import torch.optim as optim
import random
import numpy as np
import tqdm

from option import arg_parse
from dataset.ham import HAM
from torch_geometric.loader import DataLoader, DataListLoader
from model.networks import DSGPM_TP
from model.losses import TripletLoss, PosPairMSE
from utils.util import get_run_name
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler
from utils.stat import AverageMeter, FoldEpochMat
from utils.post_processing import enforce_connectivity, edge_cut_prec_recall_fscore, cg_type_prec_recall_fscore
from sklearn import metrics
from model.graph_cuts import graph_cuts

from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UndefinedMetricWarning)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(fold, epoch, train_loader, model, pos_pair_mse_criterion,
          triplet_criterion, cg_type_criterion, optimizerG, device, args):
    model.train()
    triplet_loss_meter = AverageMeter()
    pos_pair_loss_meter = AverageMeter()
    cg_type_loss_meter = AverageMeter()

    tbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)

    for i, data in tbar:
        data = data.to(device)
        model.zero_grad()

        fg_embed, fg_cg_type_pred = model(data)

        loss = 0

        # Positive pair loss
        pos_pair_loss = args.pos_pair_weight * pos_pair_mse_criterion(fg_embed, data.pos_pair_index)
        loss += pos_pair_loss
        pos_pair_loss_meter.update(pos_pair_loss.item())

        # Triplet loss (only if triplets exist)
        if torch.numel(data.triplet_index) > 0:
            triplet_loss = args.triplet_weight * triplet_criterion(fg_embed, data.triplet_index)
            loss += triplet_loss
            triplet_loss_meter.update(triplet_loss.item())

        # CG type classification loss
        cg_type_loss = args.cg_type_loss_parameter * cg_type_criterion(
            fg_cg_type_pred, data.atom_CG_types.reshape(-1).long())
        cg_type_loss_meter.update(cg_type_loss.item())
        loss += cg_type_loss

        loss.backward()
        optimizerG.step()

        tbar.set_description(
            'fold:%d [%d/%d] triplet: %.4f, pos_pair: %.4f, cg_type: %.4f'
            % (fold + 1, epoch, args.epoch, triplet_loss_meter.avg,
               pos_pair_loss_meter.avg, cg_type_loss_meter.avg))

    return triplet_loss_meter.avg, pos_pair_loss_meter.avg, cg_type_loss_meter.avg


def evaluate(fold, epoch, test_dataloader, model, device, args):
    model.eval()
    adjusted_mutual_info_meter = AverageMeter()
    edge_cut_precision_meter = AverageMeter()
    edge_cut_recall_meter = AverageMeter()
    edge_cut_f_score_meter = AverageMeter()

    type_precision_meter = AverageMeter()
    type_recall_meter = AverageMeter()
    type_f_score_meter = AverageMeter()

    tbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), dynamic_ncols=True)
    for i, data in tbar:
        data = data[0]
        num_nodes = data.x.shape[0]
        data.batch = torch.zeros(num_nodes).long()
        data = data.to(device)

        gt_hard_assigns = data.y.cpu().numpy()
        assert gt_hard_assigns.ndim == 2, \
            f"Expected data.y to be 2D [num_annotations, num_nodes], got shape {gt_hard_assigns.shape}"
        edge_index_cpu = data.edge_index.cpu().numpy()

        max_num_cg_beads = gt_hard_assigns.max(axis=1) + 1

        fg_embed, fg_cg_type_pred = model(data)
        softmax_output = F.softmax(fg_cg_type_pred, dim=1)
        predicted_cg_types_id = torch.argmax(softmax_output.cpu(), dim=1)

        for _ in range(args.test_shots):
            best_adjusted_mutual_info = -1
            best_precision, best_recall, best_f_score = 0, 0, 0
            best_precision_type, best_recall_type, best_f_score_type = 0, 0, 0

            for anno_idx, gt_hard_assign in enumerate(gt_hard_assigns):
                hard_assign, _ = graph_cuts(
                    fg_embed, data.edge_index, max_num_cg_beads[anno_idx],
                    args.bandwidth, device=device)
                try:
                    hard_assign = enforce_connectivity(hard_assign, edge_index_cpu)
                except Exception as e:
                    print(f"Warning: enforce_connectivity failed (fold {fold+1}, "
                          f"epoch {epoch}, sample {i}): {type(e).__name__}: {e}")

                precision, recall, f_score = edge_cut_prec_recall_fscore(
                    hard_assign, gt_hard_assign, edge_index_cpu)
                adjusted_mutual_info = metrics.adjusted_mutual_info_score(gt_hard_assign, hard_assign)

                best_adjusted_mutual_info = max(adjusted_mutual_info, best_adjusted_mutual_info)
                best_precision = max(precision, best_precision)
                best_recall = max(recall, best_recall)
                best_f_score = max(f_score, best_f_score)

                type_precision, type_recall, type_f_score = cg_type_prec_recall_fscore(
                    real_result=data.atom_CG_types.reshape(-1).cpu(),
                    predict_result=predicted_cg_types_id)
                best_precision_type = max(type_precision, best_precision_type)
                best_recall_type = max(type_recall, best_recall_type)
                best_f_score_type = max(type_f_score, best_f_score_type)

            adjusted_mutual_info_meter.update(best_adjusted_mutual_info)
            edge_cut_precision_meter.update(best_precision)
            edge_cut_recall_meter.update(best_recall)
            edge_cut_f_score_meter.update(best_f_score)

            type_precision_meter.update(best_precision_type)
            type_recall_meter.update(best_recall_type)
            type_f_score_meter.update(best_f_score_type)

        tbar.set_description(
            'fold:{} [{}/{}]: cg_type_prec: {:.4f}, cg_type_recall: {:.4f}, cg_type_fscore: {:.4f}'
            .format(fold + 1, epoch, args.epoch, type_precision_meter.avg,
                    type_recall_meter.avg, type_f_score_meter.avg))

    return (adjusted_mutual_info_meter.avg, edge_cut_precision_meter.avg,
            edge_cut_recall_meter.avg, edge_cut_f_score_meter.avg,
            type_precision_meter.avg, type_recall_meter.avg, type_f_score_meter.avg)


def main():
    args = arg_parse()
    assert args.ckpt is not None, '--ckpt is required'

    device = torch.device(f'cuda:{args.devices[0]}' if torch.cuda.is_available() else 'cpu')

    set_seed(42)

    train_set = HAM(data_root=args.data_root, dataset_type='train',
                    cycle_feat=args.use_cycle_feat, degree_feat=args.use_degree_feat,
                    charge_feat=args.use_charge_feat, aromatic_feat=args.use_aromatic_feat,
                    cross_validation=True, automorphism=True)
    test_set = HAM(data_root=args.data_root, dataset_type='test',
                   cycle_feat=args.use_cycle_feat, degree_feat=args.use_degree_feat,
                   charge_feat=args.use_charge_feat, aromatic_feat=args.use_aromatic_feat,
                   cross_validation=True, automorphism=True)
    assert len(train_set) == len(test_set)

    indices = list(range(len(train_set)))
    random.shuffle(indices)

    test_set_len = int(len(train_set) / args.fold)

    fold_epoch_matrix_manager = FoldEpochMat(
        args.fold, args.epoch, ['ami', 'cg_type_prec'],
        'ami', 'cut_prec', 'cut_recall', 'cut_fscore',
        'cg_type_prec', 'cg_type_recall', 'cg_type_fscore')

    for idx_fold in range(args.fold):

        print('fold [{}/{}]:'.format(idx_fold + 1, args.fold))

        test_indices = indices[idx_fold * test_set_len: (idx_fold + 1) * test_set_len]
        train_indices = list(set(indices) - set(test_indices))

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                      sampler=train_sampler, num_workers=args.num_workers,
                                      pin_memory=True)
        test_dataloader = DataListLoader(test_set, batch_size=1, num_workers=0,
                                         sampler=test_sampler, pin_memory=True)

        model = DSGPM_TP(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            use_degree_feat=args.use_degree_feat,
            use_cycle_feat=args.use_cycle_feat,
            use_charge_feat=args.use_charge_feat,
            use_aromatic_feat=args.use_aromatic_feat,
            dropout=args.dropout
        ).to(device)

        pos_pair_mse_criterion = PosPairMSE().to(device)
        triplet_criterion = TripletLoss(args.margin).to(device)
        cg_type_criterion = nn.CrossEntropyLoss().to(device)

        optimizerG = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        writer = None
        ckpt_dir = None

        if not args.debug:
            run_name = get_run_name(args.title)

            ckpt_dir = os.path.join(args.ckpt, f'{run_name}_fold_{idx_fold + 1}')
            os.makedirs(ckpt_dir, exist_ok=True)

            if args.tb_log:
                tensorboard_dir = os.path.join(args.tb_root, f'{run_name}_fold_{idx_fold + 1}')
                os.makedirs(tensorboard_dir, exist_ok=True)
                writer = SummaryWriter(tensorboard_dir)

        for e in range(1, args.epoch + 1):
            triplet_loss, pos_pair_loss, cg_type_loss = train(
                idx_fold, e, train_dataloader, model, pos_pair_mse_criterion,
                triplet_criterion, cg_type_criterion, optimizerG, device, args)

            if writer is not None:
                writer.add_scalar('triplet_loss', triplet_loss, e)
                writer.add_scalar('pos_pair_loss', pos_pair_loss, e)
                writer.add_scalar('cg_type_loss', cg_type_loss, e)

            if e % args.eval_interval == 0 and e >= args.start_eval_epoch:
                with torch.no_grad():
                    (test_ami, test_cut_prec, test_cut_recall, test_cut_fscore,
                     test_type_prec, test_type_recall, test_type_fscore) = evaluate(
                        idx_fold, e, test_dataloader, model, device, args)

                fold_epoch_matrix_manager.update(idx_fold, e - 1, {
                    'ami': test_ami,
                    'cut_prec': test_cut_prec,
                    'cut_recall': test_cut_recall,
                    'cut_fscore': test_cut_fscore,
                    'cg_type_prec': test_type_prec,
                    'cg_type_recall': test_type_recall,
                    'cg_type_fscore': test_type_fscore
                })
                best_epoch, _ = fold_epoch_matrix_manager.update_best_epoch(idx_fold)

                # Save model when current epoch is the best
                if not args.debug and best_epoch == e:
                    state_dict = (model.module.state_dict()
                                  if hasattr(model, 'module') else model.state_dict())
                    torch.save(state_dict, os.path.join(ckpt_dir, 'best_epoch.pth'))

                if writer is not None:
                    writer.add_scalar('test_ami', test_ami, e)
                    writer.add_scalar('test_cut_prec', test_cut_prec, e)
                    writer.add_scalar('test_cut_recall', test_cut_recall, e)
                    writer.add_scalar('test_cut_fscore', test_cut_fscore, e)
                    writer.add_scalar('test_type_prec', test_type_prec, e)
                    writer.add_scalar('test_type_recall', test_type_recall, e)
                    writer.add_scalar('test_type_fscore', test_type_fscore, e)

        if writer is not None:
            writer.close()

        best_epoch, epoch_metrics = fold_epoch_matrix_manager.update_best_epoch(idx_fold)
        print('\n[{}/{}] cross validation result:'.format(idx_fold + 1, args.fold))
        print(f'best_epoch: {best_epoch}')
        for met, values in epoch_metrics.items():
            if met not in ['fold', 'best_epoch']:
                print(f'{met}: {values:.4f}')
        print()

        del model, optimizerG, pos_pair_mse_criterion, triplet_criterion, cg_type_criterion
        torch.cuda.empty_cache()

    average_all_best_epoch = fold_epoch_matrix_manager.average_all_best_epoch_of_fold()
    print('\nAverage of all best epoch of fold:')
    for met, values in average_all_best_epoch.items():
        print(f'{met}: {values:.4f}')


if __name__ == '__main__':
    main()