# coding=utf-8
import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.ma.core import array
from torchvision import transforms
from network import *
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx, ImageList_strong
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

print(torch.cuda.device_count())
print(torch.cuda.is_available())

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def train_transform():
    transform_list = [
        transforms.Resize(size=(224, 224)),
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size

    three_path = f'./data/{args.dset}/{args.three_data}_list.txt'

    print(f't_dset_path={args.t_dset_path}')
    print(f'test_dset_path={args.test_dset_path}')
    print(f'AgentData={three_path}')
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    three_paths = open(three_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"],
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=True)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"],
                                      batch_size=train_bs,
                                      shuffle=False,
                                      num_workers=args.worker,
                                      drop_last=False)

    dsets["three"] = ImageList_idx(three_paths, transform=image_test())
    dset_loaders["three"] = DataLoader(dsets["three"],
                                       batch_size=train_bs,
                                       shuffle=True,
                                       num_workers=args.worker,
                                       drop_last=True)
    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def getThreeData(three_data, three_label, num=5):
    data_three2 = []
    label_three2 = []
    length = int(len(three_data) / 128)
    if length <= num:
        print('no random sample')
        start_ep = 0
    else:
        print('raddom sample')
        start_ep = np.random.randint(length - num)
    end_up = length
    print(f'length={length} start_ep={start_ep} end_up={end_up}')
    index = 0
    for ind in range(start_ep, end_up):
        if index > num:
            break
        temp_data = []
        temp_label = []
        for i in range(128):
            temp_data.append(three_data[(ind * 128) + i])
            temp_label.append(three_label[(ind * 128) + i])
        try:
            temp_data = np.asarray(temp_data)
            temp_label = np.asarray(temp_label)
            temp_data = torch.tensor(temp_data)
            temp_label = torch.tensor(temp_label)
            if len(temp_data) == 128:
                data_three2.append(temp_data)
                label_three2.append(temp_label)
            index += 1
        except Exception as e:
            print(f"Error occurred at index {index}: {e}")
            print('error1, skip')
            import traceback
            traceback.print_exc()

    return data_three2, label_three2

def FieldAlignment(epoch, three_data, three_label, optimizer, featrues_bans, netF, netB, netC):

    print('start domain alignment')

    if len(three_data) != 0:
        combined = list(zip(three_data, three_label))

        random.shuffle(combined)

        three_data, three_label = zip(*combined)
        three_data = list(three_data)
        three_label = list(three_label)

        data_three, label_three = getThreeData(three_data, three_label, 5)
        print(f'need compute{len(data_three)}')

        for ind in range(len(data_three)):
            inputs = data_three[ind]
            labels = label_three[ind]

            labels2 = []
            labels3 = []
            inputs2 = []
            for j in range(len(labels)):  
                class_data = featrues_bans[labels[j]]
                array1 = []
                for m in range(len(class_data)):
                    array1.append(class_data[m]['ft'])
                if len(array1) != 0:
                    temp1 = torch.tensor(np.asarray(array1))  # [n, 256]
                    temp1 = torch.mean(temp1, dim=0)  # [256]
                    labels2.append(temp1)
                    labels3.append(labels[j])
                    inputs2.append(inputs[j])

            labels2 = torch.tensor(np.asarray(labels2)).cuda()
            labels3 = torch.tensor(np.asarray(labels3)).cuda()
            inputs2 = torch.tensor(np.asarray(inputs2)).cuda()

            adData = inputs2.clone() 
            adData = adData.cuda()
            adData.requires_grad = True
            print(f'The{ind}-start domain shift compute')

            netF.eval()
            netB.eval()
            iter_num = 10
            for i in range(iter_num):
                output1 = netF(adData)
                output2 = netB(output1)  # [128, 256]

                classLoss = nn.MSELoss()(output2, labels2)

                adData.grad = None  
                classLoss.backward(retain_graph=True)  
                cgs = adData.grad 
                cgsView = cgs.view(cgs.shape[0], -1)
                cgsnorms = torch.norm(cgsView, dim=1) + 1e-18
                cgsView /= cgsnorms[:, np.newaxis]

                adData.data = adData.data - 0.02 * torch.sign(cgs)
                adData.data = torch.clamp(adData.data, inputs2.data - 0.2, inputs2.data + 0.2)

            netF.train()
            netB.train()
            for i in range(1):
                output1 = netF(adData)
                output2 = netB(output1)  # [128, 256]
                outputs = netC(output2)

                classLoss = nn.CrossEntropyLoss()(outputs, labels3)
                optimizer.zero_grad()
                classLoss.backward()
                optimizer.step()

        netF.eval()
        netB.eval()

    print('domain alignment finish!')


def train_target(args):
    dset_loaders = data_load(args)

    ## set base network
    if args.net[0:3] == 'res':
        print('use resnet50')
        netF = ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        netF = ViT().cuda()
    netB = feat_bottleneck(type=args.classifier,
                           feature_dim=netF.in_features,
                           bottleneck_dim=args.bottleneck).cuda()
    netC = feat_classifier(type=args.layer,
                           class_num=args.class_num,
                           bottleneck_dim=args.bottleneck).cuda()
    print(f'output_dir_src={args.output_dir_src}')
    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath), strict=False)
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()

    if args.net2 == 'res':
        print('use resnet50')
        netF2 = ResBase(res_name=args.net).cuda()
    elif args.net2 == 'vgg':
        netF2 = VGGBase(vgg_name=args.net).cuda()
    elif args.net2 == 'vit':
        print('use vit')
        netF2 = ViT().cuda()
    netB2 = feat_bottleneck(type=args.classifier,
                            feature_dim=netF2.in_features,
                            bottleneck_dim=args.bottleneck).cuda()
    netC2 = feat_classifier(type=args.layer,
                            class_num=args.class_num,
                            bottleneck_dim=args.bottleneck).cuda()

    modelpath2 = f'./ckps/source/uda/{args.dset2}/{args.three_data[0].upper()}/source_F.pt'
    print(f'AgentData path:{modelpath2}')
    netF2.load_state_dict(torch.load(modelpath2), strict=False)
    modelpath2 = f'./ckps/source/uda/{args.dset2}/{args.three_data[0].upper()}/source_B.pt'
    netB2.load_state_dict(torch.load(modelpath2))
    modelpath2 = f'./ckps/source/uda/{args.dset2}/{args.three_data[0].upper()}/source_C.pt'
    netC2.load_state_dict(torch.load(modelpath2))
    netC2.eval()

    for k, v in netC.named_parameters():
        v.requires_grad = False

    netF = nn.DataParallel(netF)
    netB = nn.DataParallel(netB)
    netC = nn.DataParallel(netC)
    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group, lr=args.lr)
    optimizer = op_copy(optimizer)

    for k, v in netC2.named_parameters():
        v.requires_grad = False

    netF2 = nn.DataParallel(netF2)
    netB2 = nn.DataParallel(netB2)
    netC2 = nn.DataParallel(netC2)
    param_group2 = []
    for k, v in netF2.named_parameters():
        if args.lr_decay1 > 0:
            param_group2 += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB2.named_parameters():
        if args.lr_decay2 > 0:
            param_group2 += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer2 = optim.SGD(param_group2, lr=args.lr)
    optimizer2 = op_copy(optimizer2)

    max_iter = args.max_epoch * len(dset_loaders["target"])  
    print('initial data')
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    print(f'sum of the samples={num_sample}')
    score_bank = torch.zeros(num_sample, args.class_num).cuda()
    # score_bank2 = torch.zeros(num_sample, args.class_num).cuda()
    featrues_bans = []
    for i in range(args.class_num):
        featrues_bans.append([])

    netF.eval()
    netB.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            indx = data[2]  
            inputs = torch.tensor(np.asarray(inputs)).cuda()
            output = netB(netF(inputs))  # [64,256]
            outputs = netC(output)  # [64, 65]
            outputs = nn.Softmax(dim=1)(outputs)  # [64, 65]
            pred = torch.argmax(outputs, dim=1)  # [64]

            score_bank[indx] = outputs.detach().clone()
          
            for i in range(len(outputs)):
                pr = outputs[i, pred[i]]
                if pr > args.delta:
                    t = {'pr': outputs[i].detach().cpu().clone(), 'ft': output[i].detach().cpu().clone()}
                    featrues_bans[pred[i]].append(t)

        three_data = []
        three_label = []
        for batchNo, (data, labels, _) in enumerate(dset_loaders["three"]):  
            for i in range(len(labels)):
                three_data.append(data[i])
                three_label.append(labels[i])
    print('initial data finish!')

    for epoch in range(args.max_epoch):
        tol_classifier_loss = 0
        ratiof = 0
        ratiot = 0
        ratioz = 0
        acc1 = 0
        if (epoch + 1) % 5 == 0:
            print('target model compute')
            FieldAlignment(epoch, three_data, three_label, optimizer, featrues_bans, netF, netB, netC)
        featrues_bans = []
        for i in range(args.class_num):
            featrues_bans.append([])
        featrues_bans2 = []
        for i in range(args.class_num):
            featrues_bans2.append([])
        for batchNo, (inputs_test_w, truth_label, tar_idx) in enumerate(loader):
            if inputs_test_w.size(0) == 1:
                continue

            if batchNo == 0:
                netF.eval()
                netB.eval()
                netF2.eval()
                netB2.eval()

                dd, all_output, labelset = obtain_label(dset_loaders['test'], netF, netB, netC, args)
                dd2, all_output2, labelset2 = obtain_label(dset_loaders['test'], netF2, netB2, netC2, args)

                netF.train()
                netB.train()
                netF2.train()
                netB2.train()

            inputs_test_w = inputs_test_w.cuda()

            lr_scheduler(optimizer, iter_num=batchNo + (epoch * len(dset_loaders["target"])), max_iter=max_iter)
            lr_scheduler(optimizer2, iter_num=batchNo + (epoch * len(dset_loaders["target"])), max_iter=max_iter)

            epoch_iter_num = 1
            if (epoch) % 4 == 0:
                epoch_iter_num = 2
            for it in range(epoch_iter_num):

                array1 = np.asarray((dd[tar_idx]))
                array2 = np.asarray((dd2[tar_idx]))
                distance1 = [min(batch) for batch in array1]
                distance2 = [min(batch) for batch in array2]
                distance1_inx = [batch.argmin(axis=0) for batch in array1]
                distance2_inx = [batch.argmin(axis=0) for batch in array2]
                pred = []
                for inx in range(len(distance1)):
                    if distance1[inx] <= (1 + epoch * args.decay) * distance2[inx]:  
                        pred.append(distance1_inx[inx])
                    else:
                        sta = False
                        for j in range(len(distance1_inx)):
                            if distance2_inx[inx] == distance1_inx[j]:
                                sta = True 
                                break
                        if sta:
                            pred.append(distance2_inx[inx])
                        else:
                            pred.append(distance1_inx[inx])
                pred = np.asarray(pred)
                try:
                    pred1 = torch.tensor(labelset[pred]).cuda()
                except:
                    print('error')
                    print(labelset.shape)
                    print(pred.shape)

                origin_pred = dd[tar_idx].argmin(axis=1)
                pred2 = torch.tensor(labelset2[dd2[tar_idx].argmin(axis=1)]).cuda()
                inputs_test_w = inputs_test_w
                features_test_w = netB(netF(inputs_test_w))
                outputs_test_w = netC(features_test_w)
                outputs_test_w = nn.Softmax(dim=1)(outputs_test_w)

                classifier_loss = nn.CrossEntropyLoss()(outputs_test_w, pred1)

                with torch.no_grad():
                    score_bank[tar_idx] = outputs_test_w.detach().clone()
                    pred_w = torch.argmax(outputs_test_w, dim=1)
                    for k in range(len(outputs_test_w)):
                        if outputs_test_w[k, pred_w[k]] > args.delta:
                            t = {'pr': outputs_test_w[k].detach().cpu().clone(),
                                 'ft': features_test_w[k].detach().cpu().clone()}
                            featrues_bans[pred_w[k]].append(t)

                optimizer.zero_grad()
                classifier_loss.backward()
                optimizer.step()

                classifier_loss = torch.nan_to_num(classifier_loss, nan=0.0)
                tol_classifier_loss += classifier_loss.item()

                # ---------------------------------------------------------------------------

                inputs_test_w2 = inputs_test_w
                features_test_w2 = netB2(netF2(inputs_test_w2))
                outputs_test_w2 = netC2(features_test_w2)
                outputs_test_w2 = nn.Softmax(dim=1)(outputs_test_w2)

                classifier_loss2 = nn.CrossEntropyLoss()(outputs_test_w2, pred2)

                optimizer2.zero_grad()
                classifier_loss2.backward()
                optimizer2.step()

            if batchNo == len(loader) - 1:
                netF.eval()
                netB.eval()
                netF2.eval()
                netB2.eval()

                if args.dset == 'VISDA-C':
                    acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                    log_str = 'Task: {}, epoch={}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, epoch, batchNo,
                                                                                          len(loader),
                                                                                          acc_s_te) + '\n' + acc_list
                else:
                    acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                    log_str = 'Task: {}, epoch={}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name,
                                                                                          epoch,
                                                                                          batchNo,
                                                                                          len(loader),
                                                                                          acc_s_te)
                    acc_s_te, _ = cal_acc(dset_loaders['test'], netF2, netB2, netC2, False)
                    log_str2 = 'Task2: {}, epoch={}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name2, epoch, batchNo,
                                                                                            len(loader), acc_s_te)

                print(log_str + '\n')
                print(log_str2 + '\n')
                args.out_file.write(log_str + '\n')
                args.out_file.flush()
                netF.train()
                netB.train()
                netF2.train()
                netB2.train()

        print(f'tol_classifier_loss={tol_classifier_loss}')

        if args.issave:
            torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
            torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
            torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    all_output_np = all_output.cpu().numpy()

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    ### all_fea: extractor feature [bs,N]
    all_fea = all_fea.float().cpu().numpy()
    # K = all_output.size(1)
    K = args.class_num
    aff = all_output.float().cpu().numpy()
    ### aff: softmax output [bs,c]
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    print(f'labelset={len(labelset)}')

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return dd, all_output_np, labelset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=18, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'DomainNet'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=False)
    parser.add_argument('--kd', type=bool, default=False)
    parser.add_argument('--se', type=bool, default=False)
    parser.add_argument('--nl', type=bool, default=False)

    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='./ckps/TS/')
    parser.add_argument('--output_src', type=str, default='./ckps/source/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        # names = ['Art', '', 'Product', 'RealWorld']
        names = ['Art', 'Product']
        args.three_data = 'Clipart'
        args.class_num = 65
    if args.dset == 'office':
        # names = ['dslr', 'amazon', 'webcam']
        names = ['dslr', 'amazon']
        args.three_data = 'webcam'
        args.class_num = 31
    if args.dset == 'DomainNet':
        # names = ['sketch', 'clipart', 'painting', 'real']
        names = ['real', 'painting']
        # names = ['sketch', 'painting']
        args.class_num = 126
        args.three_data = 'clipart'
    if args.dset == 'imageCLEF-DA':
        #names = ['i', 'p', 'c']
        names = ['p', 'i']
        args.three_data = 'c'
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()
        args.name2 = args.three_data[0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        args.delta = 0.9
        args.decay = 0.05  # office31=0.01 office-home=0.05 domainnet=0.09
        args.net2 = 'res'
        args.dset2 = 'office'
        print(names)
        print(f'initial learning rate={args.lr} decay factor={args.decay}')
        train_target(args)
