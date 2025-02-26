"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ScanObjectNNDataLoader import ScanObjectNNDataLoader
from data_utils.ScanNetDataLoader import *

# from data_utils.PointAugs import *
from utils.PointAugs import *
from data_utils.Minkowski.ModelNetVoxelLoader import *

from test_scannet import test_scannet

# really dirty import, should fix later
from model.mink_resnets import *
from model.mink_pointnet import *
from model.mink_transformer import *

from torchvision import transforms
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import random

sys.path.append('/home/zhaotianchen/project/point-transformer/pt-cls/data_utils/voxel_scannet')
from ScanNetVoxel import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

# torch.backends.cudnn.enabled=False

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pct_cls_ssg', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to load a pretrained model [default: False]')
    parser.add_argument('--use_voxel', action='store_true', default=False, help='whether to use points or voxel')
    parser.add_argument('--voxel_size', type=float, default=0., help='the voxel size')
    parser.add_argument('--dataset', type=str, default='modelnet', help='define the dataset, could be [modelnet, modelnet_voxel, scanobjnn, scannet]')
    parser.add_argument('--mode', type=str, default='train', help='define modes, [train,eval,test,export])')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', type=int, default=0, help='how many normal channels are aside from xyz')
    parser.add_argument('--num_worker', type=int, default=0, help='dataloader threads')
    parser.add_argument('--aux', type=str, default=None, help='whether to use auxiliary information to guide training, should be in [instance, seg(label), preds]')
    parser.add_argument('--pred_path', type=str, default=None, help='whether to use the prediction')
    parser.add_argument('--seed', type=int, default=2021, help= 'seed')
    return parser.parse_args()

def test(model, loader, num_class=40, log_string=None):
    # TODO: dirty fix of parsing the log_string directly in
    if "scannet" in args.dataset:
        # when using the scannet dataset val, the dataloader is an inf loader, could not use tqdm here
        dataloader_len = len(loader.dataset.scene_points_list)
    else:
        dataloader_len = len(loader)

    if "scannet" in args.dataset:
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_class)]
        total_correct_class = [0 for _ in range(num_class)]
        total_iou_deno_class = [0 for _ in range(num_class)]
    else:
        mean_correct = []
        class_acc = np.zeros((num_class,3))

    for j, data in tqdm(enumerate(loader), total=dataloader_len):
        if "modelnet" in args.dataset:
            points, target = data
            target = target[:, 0]
        elif "scanobjnn" in args.dataset:
            points, target, _ = data
        elif "scannet" in args.dataset:
            points, target, sample_weight = data
            points = points.float()
            sampled_weight = sample_weight.cuda()

        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        classifier = model.eval()
        criterion = model.loss
        pred = classifier(points)

        loss_list = []
        if "scannet" in args.dataset:
            loss = criterion(pred, target.long(), sample_weight)
        else:
            loss = criterion(pred, target.long())
        loss_list.append(loss.item())

        if "scannet" in args.dataset:
            pred_choice = torch.argmax(pred, dim=2).cpu().numpy()  # B,N
            target = target.cpu().numpy()
            sample_weight = sample_weight.cpu().numpy()
            correct = np.sum((pred_choice == target) & (target > 0) & (sample_weight>0))
            total_correct += correct
            total_seen += np.sum((target>0) & (sample_weight>0))

            for l in range(num_class):
                total_seen_class[l] += np.sum((target==l) & (sample_weight>0))
                total_correct_class[l] += np.sum((pred_choice==l) & (target==l) & (sample_weight>0))
                total_iou_deno_class[l] += np.sum(((pred_choice==l) | (target==l)) & (sample_weight>0) & (target>0))
        else:
            pred_choice = pred.data.max(1)[1]
            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
                class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
                class_acc[cat,1]+=1
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item()/float(points.size()[0]))

    if "scannet" in args.dataset:
        IoU = np.array(total_correct_class[1:])/(np.array(total_iou_deno_class[1:],dtype=np.float)+1e-6)
        log_string('eval point avg class IoU: %f' % (np.mean(IoU)))
        IoU_Class = 'Each Class IoU:::\n'
        # for i in range(IoU.shape[0]):
            # log_string('Class %d : %.4fA'%(i+1, IoU[i]))
        mIoU = np.mean(IoU)
        log_string('eval ss: %f'% (np.mean(loss_list)))
        log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
        log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
        log_string('Eval loss: %f '% np.mean(loss_list))
        log_string('Eval mIoU %f' % mIoU)
        log_string('Eval oA: %f' % (total_correct / float(total_seen)))
        log_string('Eval mA: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))

        return mIoU
    else:
        class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
        if "modelnet" in args.dataset:
            class_acc = np.mean(class_acc[:,2])
        elif "scanobjnn" in args.dataset:
            class_acc = class_acc[:,2]
        instance_acc = np.mean(mean_correct)
        if "scanobjnn" in args.dataset:
            class_acc = np.mean(class_acc)

        return instance_acc, class_acc

def main(args):

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''SET THE SEED'''
    setup_seed(args.seed)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    # log_dir = experiment_dir.joinpath('logs/')
    log_dir = experiment_dir.joinpath('./')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA TYPE'''
    if args.use_voxel:
        assert "voxel" in args.dataset
        assert "mink" in args.model
        assert args.voxel_size > 0



    '''AUX SUPERVISION TYPE'''
    if args.aux == "pred":
        assert args.pred_path is not None

    if args.pred_path is not None:
        assert args.aux == "pred"

    args.with_pred = None
    args.with_instance = False
    args.with_seg = False
    if args.aux is not None:
        args.with_aux = True
        assert "scannet" in args.dataset
        if args.aux == "pred":
            args.with_pred = args.pred_path
        elif args.aux == "instance":
            args.with_instance = True
        elif args.aux == "seg":
            args.with_seg = True
        else:
            raise NotImplementedError
    else:
        args.with_aux = False



    '''DATA LOADING'''
    if "modelnet" in args.dataset:
        '''
        the modelnet 40 loading, support both the point & ME-point Ver.
        '''
        if not "voxel" in args.dataset:
            log_string('Load dataset {}'.format(args.dataset))
            num_class = 40
            DATA_PATH = './data/modelnet40_normal_resampled/'
            TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',
                                                             normal_channel=args.normal, apply_aug=True)
            TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal)

            trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
            testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
        else:
            assert args.dataset == 'modelnet_voxel'
            '''
            use the modelnet example dataloader from the ME-engine
            however, it seems still is point-based, retrun point features，
            and feed in the TenesorField, not really the voxel-modelnet
            '''
            log_string('Load dataset {}'.format(args.dataset))
            num_class = 40
            DATA_PATH = './data/modelnet40_ply_hdf5_2048'

            trainset = ModelNet40H5(
                phase = "train",
                transform=CoordinateTransformation(trans=0.2),
                data_root = DATA_PATH,
            )
            testset = ModelNet40H5(
                phase = "test",
                transform=None,  # no transform for test
                data_root = DATA_PATH,
            )

            trainDataLoader = DataLoader(
                trainset,
                num_workers=args.num_worker,
                shuffle=True,
                batch_size=args.batch_size,
                collate_fn = minkowski_collate_fn,
                pin_memory=True,
            )

            testDataLoader = DataLoader(
                testset,
                num_workers=args.num_worker,
                shuffle=False,
                batch_size=args.batch_size,
                collate_fn = minkowski_collate_fn,
                pin_memory=True,
            )

    elif args.dataset == "scanobjnn":
        log_string('Load dataset {}'.format(args.dataset))
        num_class = 15
        DATA_PATH = './data/scanobjnn/main_split_nobg'
        TRAIN_DATASET = ScanObjectNNDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',
                                                         normal_channel=args.normal)
        TEST_DATASET = ScanObjectNNDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                        normal_channel=args.normal)
        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    elif "scannet" in args.dataset:
        num_class = 21
        if not "voxel" in args.dataset:
            if args.mode == "train":
                trainset = ScannetDataset(root='./data/scannet_v2/scannet_pickles', npoints=args.num_point, split='train', with_seg=args.with_seg, with_instance=args.with_instance, with_pred=args.pred_path)
                trainDataLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=True)
            if args.mode == 'export':
                final_trainset = ScannetDatasetWholeScene_evaluation(root='./data/scannet_v2/scannet_pickles', scene_list_dir='./data/scannet_v2/metadata',split='train',block_points=args.num_point, with_rgb=True, with_norm=True,\
                                                                     with_seg=args.with_seg, with_instance=args.with_instance, with_pred=args.pred_path, delta=2.0)
                final_train_loader = torch.utils.data.DataLoader(final_trainset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)


            final_testset = ScannetDatasetWholeScene_evaluation(root='./data/scannet_v2/scannet_pickles', scene_list_dir='./data/scannet_v2/metadata',split='eval',block_points=args.num_point, with_rgb=True, with_norm=True, \
                                                                with_seg=args.with_seg, with_instance=args.with_instance, with_pred=args.pred_path, delta=1.0) # DEBUG: change to 1.0 to axquire proper
            final_test_loader = torch.utils.data.DataLoader(final_testset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

            # generate the trainset as whole_dataset for export
        else:
            trainDataLoader = initialize_data_loader(
                    DatasetClass=ScannetSparseVoxelizationDataset,
                    data_root='data/scannet_v2/scannet_pickles',
                    phase="train",
                    threads=4, # num-workers
                    shuffle=True,
                    repeat=False,
                    augment_data=True,
                    batch_size=16,
                    limit_numpoints=1200000,
                )

            # TODO: the testloader

    else:
        raise NotImplementedError

    '''MODEL LOADING'''
    # copy files
    if args.mode == "train":
        if not os.path.exists(os.path.join(str(experiment_dir),'model')):
            os.mkdir(os.path.join(str(experiment_dir),'model'))
        for filename in os.listdir('./model'):
            if ".py" in filename:
                shutil.copy(os.path.join("./model", filename), os.path.join(str(experiment_dir),'model'))
        shutil.copy("./train_cls.py", str(experiment_dir))

    if "mink" not in args.model:
        # no use mink-net
        if "seg" in args.model:
            N = args.num_point
        else:
            N = args.num_point
        MODEL = importlib.import_module(args.model)
        classifier = MODEL.get_model(num_class,normal_channel=args.normal, N=N).cuda()
        criterion = MODEL.get_loss().cuda()
        classifier.loss = criterion
    else:
        '''
        The Voxel-based Networks based on the MinkowskiEngine
        '''
        # TODO: should align with above, using importlib.import_module, maybe fix later
        # classifier = ResNet14(in_channels=3, out_channels=num_class, D=3)  # D is the conv spatial dimension, 3 menas 3-d shapes
        if "pointnet" in args.model:
            classifier = MinkowskiPointNet(in_channel=3, out_channel=41, embedding_channel=1024,dimension=3).cuda()
        elif "trans" in args.model:
            classifier = MinkowskiTransformer(in_channel=3, out_channel=41, num_class=num_class, embedding_channel=1024, dimension=3).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        classifier.loss = criterion


    '''Loading existing ckpt'''
    try:
        if args.pretrain:
            # FIXME: currently only loading the best_model.pth, should support string, maybe latter
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model')
            start_epoch = 0
        else:
            start_epoch = 0
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), \
                                    lr=args.learning_rate, momentum=0.9,\
                                    weight_decay=args.decay_rate)
    else:
        raise NotImplementedError

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    # Use MultiStepLR as in paper, decay by 10 at [120, 160]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    # FIXME:  for scannet, now using the cosine anneal
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=0.0)

    global_epoch = 0
    global_step = 0
    if "scannet" in args.dataset:
        best_mIoU = 0.0
    else:
        best_instance_acc = 0.0
        best_class_acc = 0.0
    mean_correct = []

    # only run for one epoch on the eval-only mode
    if args.mode == "eval" or args.mode == "export":
        assert args.pretrain
        start_epoch = 0
        args.epoch = 1

    # '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        log_string('Cur LR: {:.5f}'.format(optimizer.param_groups[0]['lr']))
        # when eval only, skip the traininig part

        if args.mode == "train":
            '''The main training-loop'''
            for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
                if not args.use_voxel:
                    if "modelnet" in args.dataset:
                        # use points, normal unpacking
                        points, target = data
                        points = points.data.numpy()
                        points = provider.random_point_dropout(points)
                        # points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
                        # points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
                        points = torch.Tensor(points)
                        target = target[:, 0]

                        points = points.transpose(2, 1)
                        points, target = points.cuda(), target.cuda()
                    elif "scanobjnn" in args.dataset:
                        points, target, mask = data
                        points = points.data.numpy()
                        # TODO: move the aug in the dataset but not here
                        # points = provider.random_point_dropout(points)
                        points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
                        points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
                        points = torch.Tensor(points)
                        points = points.transpose(2, 1)
                        points, target = points.cuda(), target.cuda()
                    elif "scannet" in args.dataset:
                        # TODO: fiil the scannet loading here
                        # TODO: maybe implement the grad-accmu/or simply not
                        if args.aux is not None:
                            points, target, sample_weight, aux = data
                            points, target, sample_weight, aux = points.float().transpose(1,2).cuda(), target.cuda(), sample_weight.cuda(), aux.cuda()
                        else:
                            points, target, sample_weight = data
                            points, target, sample_weight = points.float().transpose(1,2).cuda(), target.cuda(), sample_weight.cuda()

                else:
                    if "modelnet" in args.dataset:
                        # use voxel
                        # points = create_input_batch(data, True, 'cuda', quantization_size=args.voxel_size) 
                        data['coordinates'][:,1:] = data['coordinates'][:,1:]/args.voxel_size
                        points = ME.TensorField(coordinates=(data['coordinates'].cuda()), features=data['features'].cuda())
                        target = data['labels'].cuda()
                    elif "scannet" in args.dataset:
                        dat = ME.SparseTensor(features=data[1],coordinates=data[0]).cuda()
                        target = data[2].cuda()

                optimizer.zero_grad()

                '''save the intermediate attention map'''
                # WANINIG: DISABLED FOR NOW!!!
                SAVE_INTERVAL = 50
                NUM_PER_EPOCH = 1

                if (epoch+1) % SAVE_INTERVAL == 0:
                    if batch_id < NUM_PER_EPOCH:
                        classifier.save_flag = True
                    elif batch_id == NUM_PER_EPOCH:
                        intermediate_dict = classifier.save_intermediate()
                        intermediate_path = os.path.join(experiment_dir, "attn")
                        if not os.path.exists(intermediate_path):
                            os.mkdir(intermediate_path)
                        torch.save(intermediate_dict, os.path.join(intermediate_path, "epoch_{}".format(epoch)))
                        log_string('Saved Intermediate at {}'.format(epoch))
                    else:
                        classifier.save_flag = False
                else:
                    classifier.save_flag = False

                classifier = classifier.train()
                # when with-instance, use instance label to guide the point-transformer training
                if args.aux is not None:
                    pred = classifier(points, aux)
                else:
                    pred = classifier(points)
                # if use_voxel, get the feature from the SparseTensor
                if args.use_voxel:
                    pred = pred.F
                if 'scannet' in args.dataset:
                    loss = criterion(pred, target.long(), sample_weight)
                else:
                    loss = criterion(pred, target.long())
                loss.backward()
                optimizer.step()
                global_step += 1

                if "scannet" in args.dataset:
                    pred_choice = torch.argmax(pred, dim=2).cpu().numpy()  # B,N
                    target = target.cpu().numpy()
                    correct = np.sum(pred_choice == target)
                    mean_correct.append(correct / pred_choice.size)
                else:
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.long().data).cpu().sum()
                    mean_correct.append(correct.item() / float(points.size()[0]))

            train_instance_acc = np.mean(mean_correct)
            log_string('Train Instance Accuracy: %f' % train_instance_acc)

            '''TEST'''
            if not "scannet" in args.dataset:
                # WARNING: Temporarily disable eval for scannet for now, just test at last
                if (epoch+1) % 20 == 0:
                    with torch.no_grad():
                        returned_metric = test(classifier.eval(), testDataLoader, num_class=num_class, log_string=log_string)

                    if 'scannet' in args.dataset:
                        mIoU = returned_metric
                        if (mIoU >= best_mIoU):
                            best_mIoU = mIoU
                            best_epoch = epoch + 1

                        if (mIoU >= best_mIoU):
                            logger.info('Save model...')
                            savepath = str(checkpoints_dir) + '/best_model.pth'
                            log_string('Saving at %s'% savepath)
                            state = {
                                'epoch': best_epoch,
                                'mIoU': mIoU,
                                'model_state_dict': classifier.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                            }
                            torch.save(state, savepath)
                    else:
                        instance_acc, class_acc = returned_metric

                        if (instance_acc >= best_instance_acc):
                            best_instance_acc = instance_acc
                            best_epoch = epoch + 1

                        if (class_acc >= best_class_acc):
                            best_class_acc = class_acc

                        log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
                        log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

                        if (instance_acc >= best_instance_acc):
                            logger.info('Save model...')
                            savepath = str(checkpoints_dir) + '/best_model.pth'
                            log_string('Saving at %s'% savepath)
                            state = {
                                'epoch': best_epoch,
                                'instance_acc': instance_acc,
                                'class_acc': class_acc,
                                'model_state_dict': classifier.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                            }
                            torch.save(state, savepath)

        global_epoch += 1

    # final save of the model
    logger.info('Save model...')
    savepath = str(checkpoints_dir) + '/final_model.pth'
    log_string('Saving at %s'% savepath)
    state = {
        'epoch': global_epoch,
        # 'mIoU': mIoU,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)


    # for the scannet dataset, test at last
    if args.dataset == 'scannet':
        if not os.path.exists(os.path.join(str(experiment_dir),'pred')):
            os.mkdir(os.path.join(str(experiment_dir),'pred'))
        if args.mode == "export":
            test_scannet(args, classifier.eval(), final_test_loader, log_string, with_aux=args.with_aux, save_dir=os.path.join(str(experiment_dir),'pred'), split='eval')
            test_scannet(args, classifier.eval(), final_train_loader, log_string, with_aux=args.with_aux, save_dir=os.path.join(str(experiment_dir),'pred'), split='train')
        else:
            test_scannet(args, classifier.eval(), final_test_loader, log_string, with_aux=args.with_aux, split='eval')

    # final save of the model
    logger.info('Save model...')
    savepath = str(checkpoints_dir) + '/best_model.pth'
    log_string('Saving at %s'% savepath)
    state = {
        'epoch': global_epoch,
        # 'mIoU': mIoU,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)


    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
