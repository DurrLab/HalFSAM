from __future__ import print_function

import argparse
import os
import time
from datetime import timedelta

import cv2 
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.HalFSAM import build_HalFSAM


from datasets.datasets_video import DATASET_NAMES, dataset_info, VideoDataset
from datasets.datasets_orig import TestDatasetOutput
from misc.losses import *
from misc.utils import (image_normalization, save_image_batch_to_disk,
                   visualize_result,count_parameters)
#Temporal Consistency
from misc.utils.metric import compute_temporal_consistency_batch, EdgeDetectionMetrics
from tensorboardX import SummaryWriter

def train_one_epoch(rank, epoch, dataloader, model, criterion, optimizer, device,
                    log_interval_vis, tb_writer, args=None):
    imgs_res_folder = os.path.join(args.output_dir, 'current_res')
    if rank == 0:
        os.makedirs(imgs_res_folder,exist_ok=True)

    # Put model in training mode
    model.train()
    l_weight = [0.7, 1.1, 1.1, 0.7, 1.5] # for SAM

    loss_avg =[]
    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'][:1].to(device)  # B x T x C x H x W
        labels = sample_batched['labels'][0].to(device)  # T x 1 x H x W
        
        # Zero gradients before forward pass
        optimizer.zero_grad()
        
        # Forward pass
        preds_list = model(images)
        
        # Compute loss
        loss = sum([criterion(preds, labels, l_w) for preds, l_w in zip(preds_list, l_weight)]) # bdcn_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # Step optimizer
        optimizer.step()
        
        loss_avg.append(loss.item())
        if epoch==0 and (batch_id==100 and tb_writer is not None) and rank==0:
            tmp_loss = np.array(loss_avg).mean()
            tb_writer.add_scalar('loss', tmp_loss,epoch)

        if batch_id % 5 == 0 and rank==0:
            print(time.ctime(), 'Epoch: {0} Sample {1}/{2} Loss: {3}'
                  .format(epoch, batch_id, len(dataloader), loss.item()))
        if batch_id % log_interval_vis == 0 and rank==0:
            for fid in range(images.size(1)):
                res_data = []

                img = images.cpu().numpy()
                res_data.append(img[0, fid])

                ed_gt = labels.cpu().numpy()
                res_data.append(ed_gt[fid])

                # tmp_pred = tmp_preds[2,...]
                for i in range(len(preds_list)):
                    tmp = preds_list[i]
                    tmp = tmp[fid]
                    # print(tmp.shape)
                    tmp = torch.sigmoid(tmp).unsqueeze(dim=0)
                    tmp = tmp.cpu().detach().numpy()
                    res_data.append(tmp)

                vis_imgs = visualize_result(res_data, arg=args)
                del tmp, res_data

                vis_imgs = cv2.resize(vis_imgs,
                                    (int(vis_imgs.shape[1]*0.8), int(vis_imgs.shape[0]*0.8)))
                img_test = 'Epoch: {0} Sample {1}/{2} Loss: {3}' \
                    .format(epoch, batch_id, len(dataloader), loss.item())

                BLACK = (0, 0, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_size = 1.1
                font_color = BLACK
                font_thickness = 2
                x, y = 30, 30
                vis_imgs = cv2.putText(vis_imgs,
                                    img_test,
                                    (x, y),
                                    font, font_size, font_color, font_thickness, cv2.LINE_AA)
                cv2.imwrite(os.path.join(imgs_res_folder, f'results_{fid:d}.png'), vis_imgs)
    loss_avg = np.array(loss_avg).mean()
    return loss_avg

def maskEdges(edges, boundary=25, black_border=55):
    # Create border mask once per batch
    if len(edges.shape) == 4:
        B, C, H, W = edges.shape
    else:
        B, T, C, H, W = edges.shape
    border_mask = np.ones((H, W), dtype=bool)
    border_mask[:black_border+boundary, :] = False
    border_mask[-(black_border+boundary):, :] = False
    border_mask[:, :boundary] = False
    border_mask[:, -boundary:] = False
    border_mask = torch.from_numpy(border_mask).to(edges.device)
    edges = edges * border_mask
    return edges

def validate_one_epoch(epoch, dataloader, model, device, output_dir, arg=None):
    # XXX This is not really validation, but testing
    start = time.time()
    # Put model in eval mode
    model.eval()

    # Metric computation
    # Add IoU and Temporal Consistency metrics using utils.metric functions
    temporal_consistency = 0.0
    gt_thresh = 0.2
    valid_batch = 0
    metrics = EdgeDetectionMetrics(n_thresholds=50).to(device)

    print("Validation Started")
    with torch.no_grad():
        for i, sample_batched in enumerate(dataloader):
            images = sample_batched['images'][0].to(device)
            # labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            labels = sample_batched['labels'][0].to(device)
            flows = sample_batched['flows'][0].to(device)
            preds = model(images)
            
            # Post process predicitons and labels
            # Extact final prediction and clamp values
            preds_final = torch.sigmoid(preds[-1].float()).clamp(0, 1)
            # Binarize the gt labels and convert 
            labels = (labels > gt_thresh).int()
            # For metrics computation
            # Apply mask to preds & labels
            labels = maskEdges(labels)
            preds_final = maskEdges(preds_final)
            # flows = maskEdges(flows)

            # Flatten file names
            file_names_flat = [fn for batch in file_names for fn in batch]  # Shape: [B*T]
            # Compute Temporal Consistency
            # temporal_consistency += compute_temporal_consistency_batch(preds_final, flows, edgeBinaryThreshold =0.2)

            # Compute Overlap Metrics
            # Convert 5d to 4d tensors
            # B, T, C, H, W = labels.shape
            # labels = labels.view(B*T, C, H, W)
            # preds_final = preds_final.view(B*T, C, H, W)

            # Check if the labels contain any edges
            # Check for images in the batch with non-zero edges
            edge_sums = torch.sum(labels, dim=(1, 2, 3))  # Sum along C, H, W dimensions
            valid_indices = edge_sums > 0

            # Filter out images with no edges from both labels and preds_final
            labels = labels[valid_indices]
            preds_final = preds_final[valid_indices]

            # Update metrics
            if len(labels) > 0:
                metrics.update(preds_final, labels)
                valid_batch += 1

            save_image_batch_to_disk(preds[-1],
                                     output_dir,
                                     file_names_flat,img_shape=image_shape,
                                     arg=arg)
            torch.cuda.empty_cache()

        # Compute the average IoU and Temporal Consistency
        # temporal_consistency /= valid_batch
        results = metrics.compute()
        print(f"Epoch: {epoch} ODS: {results['ODS']:.4f}, OIS: {results['OIS']:.4f}, AP: {results['AP']:.4f}")
        # Print the time taken to validate
        print(f'Time taken to validate: {time.time() - start:.4f} seconds')        

def test_metrics(epoch, dataloader, device='cuda'):
    # Metric computation
    # Add IoU and Temporal Consistency metrics using utils.metric functions
    temporal_consistency = 0.0
    gt_thresh = 0.2
    valid_batch = 0
    metrics = EdgeDetectionMetrics(n_thresholds=50).to(device)

    total_duration = []
    for batch_id, sample_batched in enumerate(dataloader):
        preds_final = sample_batched['images'][0].to(device)
        labels = sample_batched['labels'][0].to(device)
        file_names = sample_batched['file_names']
        flows = sample_batched['flows'][0].to(device)
        image_shape = sample_batched['image_shape']
        # images = images[:, [2, 1, 0], :, :]

        # ipdb.set_trace()
        # Apply erosion to preds_final
        # convert to numpy
        # iterateover the batch
        for i in range(preds_final.shape[0]):
            preds = preds_final[i].cpu().numpy()
            # Apply erosion kernel
            # Define kernel (structuring element)
            kernel = np.ones((3, 3), np.uint8)
            # Apply erosion
            preds = cv2.erode(preds, kernel, iterations=1)
            # Convert back to tensor
            preds_final[i] = torch.from_numpy(preds).to(device)
        
        
        # Post process predicitons and labels
        # Extact final prediction and clamp values
        # preds_final = torch.sigmoid(preds.float()).clamp(0, 1)
        # Binarize the gt labels and convert 
        labels = (labels > gt_thresh).int()
        # For metrics computation
        # Apply mask to preds & labels
        labels = maskEdges(labels)
        preds_final = maskEdges(preds_final)
        flows = maskEdges(flows)

        # Flatten file names
        file_names_flat = [fn for batch in file_names for fn in batch]  # Shape: [B*T]
        # Compute Temporal Consistency
        temporal_consistency += compute_temporal_consistency_batch(preds_final, flows, edgeBinaryThreshold =0.4)

        # Check if the labels contain any edges
        # Check for images in the batch with non-zero edges
        edge_sums = torch.sum(labels, dim=(1, 2, 3))  # Sum along C, H, W dimensions
        valid_indices = edge_sums > 0

        # Filter out images with no edges from both labels and preds_final
        labels = labels[valid_indices]
        preds_final = preds_final[valid_indices]

        # Update metrics
        if len(labels) > 0:
            metrics.update(preds_final, labels)
            valid_batch += 1

        # Print batch_id
        if batch_id % 20 == 0:
            print(f"Epoch: {epoch} Batch: {batch_id}/{len(dataloader)}")

    # Compute the average IoU and Temporal Consistency
    temporal_consistency /= valid_batch
    results = metrics.compute()
    print(f"Epoch: {epoch} ODS: {results['ODS']:.4f}, OIS: {results['OIS']:.4f}, AP: {results['AP']:.4f}, Temporal Consistency: {temporal_consistency:.4f}")
    # Save results to text file
    with open('memSAM_results.txt', 'a') as f:
        f.write(f"Epoch: {epoch} ODS: {results['ODS']:.4f}, OIS: {results['OIS']:.4f}, AP: {results['AP']:.4f}, Temporal Consistency: {temporal_consistency:.4f} \n")
        
def test(checkpoint_path, dataloader, model, device, output_dir, args):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    temporal_consistency = 0.0
    gt_thresh = 0.2
    valid_batch = 0
    metrics = EdgeDetectionMetrics(n_thresholds=50).to(device)


    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            # OLD
            # images = sample_batched['images'].to(device)
            # if not args.test_data == "CLASSIC":
            #     labels = sample_batched['labels'].to(device)
            # file_names = sample_batched['file_names']
            # image_shape = sample_batched['image_shape']
            # # print(f"input tensor shape: {images.shape}")
            # # images = images[:, [2, 1, 0], :, :]

            # Metrics Computation
            images = sample_batched['images'][0].to(device)
            # labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            labels = sample_batched['labels'][0].to(device)
            flows = sample_batched['flows'][0].to(device)
            preds = model(images)

            # Flatten file names
            file_names_flat = [fn for batch in file_names for fn in batch]  # Shape: [B*T]

             # Post process predicitons and labels
            # Extact final prediction and clamp values
            preds_final = torch.sigmoid(preds[-1].float()).clamp(0, 1)
            # Binarize the gt labels and convert 
            labels = (labels > gt_thresh).int()
            # For metrics computation
            # Apply mask to preds & labels
            labels = maskEdges(labels)
            preds_final = maskEdges(preds_final)
            flows = maskEdges(flows)

            # Check if the labels contain any edges
            # Check for images in the batch with non-zero edges
            edge_sums = torch.sum(labels, dim=(1, 2, 3))  # Sum along C, H, W dimensions
            valid_indices = edge_sums > 0

            # Filter out images with no edges from both labels and preds_final
            labels = labels[valid_indices]
            preds_final = preds_final[valid_indices]

            # Update metrics
            if len(labels) > 0:
                metrics.update(preds_final, labels)
                valid_batch += 1

            # Flatten file names
            file_names_flat = [fn for batch in file_names for fn in batch]  # Shape: [B*T]
            # Compute Temporal Consistency
            temporal_consistency += compute_temporal_consistency_batch(preds_final, flows, edgeBinaryThreshold =0.2)


            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            preds = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)
            if batch_id % 1000 == 0:
                save_image_batch_to_disk(preds,
                                        output_dir,
                                        file_names_flat,
                                        image_shape,
                                        arg=args)
            torch.cuda.empty_cache()

    total_duration = np.sum(np.array(total_duration))
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("FPS: %f.4" % (len(dataloader)/total_duration))
    # Compute the average IoU and Temporal Consistency
    temporal_consistency /= valid_batch
    results = metrics.compute()
    print(f"ODS: {results['ODS']:.4f}, OIS: {results['OIS']:.4f}, AP: {results['AP']:.4f}, Temporal Consistency: {temporal_consistency:.4f}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='HalFSAM trainer.')
    parser.add_argument('--choose_test_data',
                        type=int,
                        default=-1,
                        help='Already set the dataset for testing choice: 0 - 8')
    is_testing =False#  current test -352-SM-NewGT-2AugmenPublish
    is_eval = False # Use for temporal consistency evalutaion
    
    # Training settings
    TRAIN_DATA = DATASET_NAMES[parser.parse_args().choose_test_data] #  ['BIPED', 'MDBD', 'CLASSIC', 'C3VDv2']
    train_inf = dataset_info(TRAIN_DATA)
    train_dir = train_inf['data_dir']
    
    # Test settings
    TEST_DATA = DATASET_NAMES[parser.parse_args().choose_test_data]
    test_inf = dataset_info(TEST_DATA)
    test_dir = test_inf['data_dir']

    # Data parameters
    parser.add_argument('--Encoder',
                        type=str,
                        default='SAM',
                        help='Only for image normalization. [SAM, DexiNed]')
    
    parser.add_argument('--input_dir',
                        type=str,
                        default=train_dir,
                        help='the path to the directory with the input data.')
    parser.add_argument('--input_val_dir',
                        type=str,
                        default=test_dir,
                        help='the path to the directory with the input data for validation.')
    parser.add_argument('--train_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TRAIN_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--train_list',
                        type=str,
                        default=train_inf['train_list'],
                        help='Dataset sample indices list.')
    parser.add_argument('--test_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TEST_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--test_list',
                        type=str,
                        default=test_inf['test_list'],
                        help='Dataset sample indices list.')
    parser.add_argument('--mean_pixel_values',
                        default=[103.939,116.779,123.68, 137.86],
                        type=float)  # [103.939,116.779,123.68] [104.00699, 116.66877, 122.67892]
    parser.add_argument('--img_width',
                        type=int,
                        default=512,
                        help='Image width for training.') # BIPED 400 / MDBD 480 / C3VDv2 512
    parser.add_argument('--img_height',
                        type=int,
                        default=512,
                        help='Image height for training.')
    parser.add_argument('--test_img_width',
                        type=int,
                        default=test_inf['img_width'],
                        help='Image width for testing.')
    parser.add_argument('--test_img_height',
                        type=int,
                        default=test_inf['img_height'],
                        help='Image height for testing.')
    
    parser.add_argument('--output_dir',
                        type=str,
                        default='checkpoints_HalFSAM/',
                        help='the path to output the results.')
    parser.add_argument('--res_dir',
                        type=str,
                        default='result',
                        help='Result directory')
    
    parser.add_argument('--resume',
                        type=bool,
                        default=False,
                        help='use previous trained data')
    parser.add_argument('--checkpoint_data',
                        type=str,
                        default='checkpoints_HalFSAM/C3VDv2/23/23_model.pth',
                        help='Checkpoint path from which to restore model weights from.')
    
    parser.add_argument('--is_testing',type=bool,
                        default=is_testing,
                        help='Script in testing mode.')
    parser.add_argument('--is_eval',type=bool,
                        default=is_eval,
                        help='Script in temporal consistency evaluation mode.')
    
    parser.add_argument('--log_interval_vis',
                        type=int,
                        default=20,
                        help='The number of batches to wait before printing test predictions.')
    parser.add_argument('--epochs',
                        type=int,
                        default=40,
                        metavar='N',
                        help='Number of training epochs (default: 25).')
    parser.add_argument('--lr',
                        default=1e-4,
                        type=float,
                        help='Initial learning rate.')
    parser.add_argument('--wd',
                        type=float,
                        default=1e-8,
                        metavar='WD',
                        help='weight decay (Good 1e-8) in TF1=0') # 1e-8 -> BIRND/MDBD, 0.0 -> BIPED
    parser.add_argument('--adjust_lr',
                        default=[25],
                        type=int,
                        help='Learning rate step size.') #[5,10]BIRND [10,15]BIPED/BRIND
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        metavar='B',
                        help='the mini-batch size (default: 8)')
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        help='The number of workers for the dataloaders.')
    parser.add_argument('--tensorboard',type=bool,
                        default=False,
                        help='Use Tensorboard for logging.'),
    parser.add_argument('--channel_swap',
                        default=[2, 1, 0],
                        type=int)
    parser.add_argument('--crop_img',
                        default=False,
                        type=bool,
                        help='If true crop training images, else resize images to match image width and height.')
    args = parser.parse_args()
    return args

def main(gpu, args):
    """Main function."""
    # Add max_grad_norm to args
    args.max_grad_norm = 1.0
    rank = args.node_rank * args.gpus + gpu
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size,
                            rank=rank,
                            timeout=timedelta(seconds=1000))

    # Tensorboard summary writer
    tb_writer = None
    training_dir = os.path.join(args.output_dir,args.train_data)
    if rank ==0:
        os.makedirs(training_dir,exist_ok=True)
        
    if args.tensorboard and not args.is_testing:
        #from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather
        tb_writer = SummaryWriter(log_dir=training_dir)
        # saving Model training settings
        training_notes = ['DexiNed, Xavier Normal Init, LR= ' + str(args.lr) + ' WD= '
                          + str(args.wd) + ' image size = ' + str(args.img_width)
                          + ' adjust LR='+ str(args.adjust_lr) + ' Loss Function= BDCNloss2. '
                          +'Trained on> '+args.train_data+' Tested on> '
                          +args.test_data+' Batch size= '+str(args.batch_size)+' '+str(time.asctime())]
        info_txt = open(os.path.join(training_dir, 'training_settings.txt'), 'w')
        info_txt.write(str(training_notes))
        info_txt.close()


    # Instantiate model and move it to the computing device
    # load adapted SAM encoder
    config_name = "HalFSAM.yaml"
    encoder_ckpt_path = 'checkpoints_SAM_adapt/pretrained.pth'
    model = build_HalFSAM(config_name, encoder_ckpt_path)
    
    # Move model to device first
    model = model.to(rank)
    # Convert to SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Initialize DDP with bucket_cap_mb parameter
    model = DDP(model, 
                device_ids=[rank],
                find_unused_parameters=True,
                broadcast_buffers=True,
                bucket_cap_mb=25)  # Try a smaller bucket size
    device = model.module.device
    
    # initialize training
    ini_epoch =0
    if not args.is_testing:
        if args.resume:
            checkpoint_path = args.checkpoint_data
            ini_epoch= int(checkpoint_path.split('/')[-2])
            result = model.module.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device), strict=False)
            if rank == 0:
                print('Training restarted from> ',checkpoint_path)

        # setup training data
        dataset_train = VideoDataset(data_root=args.input_dir,
                                     split_list = args.train_list,
                                     img_height=args.img_height,
                                     img_width=args.img_width,
                                     num_frames= 6,
                                     args=args,
        )
        sampler_train = DistributedSampler(dataset_train, num_replicas=args.world_size, rank=rank, shuffle=True)
        dataloader_train = DataLoader(dataset_train,
                                    batch_size=args.batch_size // args.world_size,
                                    pin_memory=True,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    sampler=sampler_train,
                                    )
    
    dataset_val = TestDatasetOutput(args.input_val_dir,
                            test_data=args.test_data,
                            img_width=args.test_img_width,
                            img_height=args.test_img_height,
                            num_frames= 6,
                            mean_bgr=args.mean_pixel_values[0:3] if len(
                                args.mean_pixel_values) == 4 else args.mean_pixel_values,
                            test_list=args.test_list, arg=args,
                            size_limit=None)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)
    
    # Print size of datasets
    if rank == 0:
        print(f"Training dataset size: {len(dataset_train)}")
        print(f"Validation dataset size: {len(dataset_val)}")
    
    # Testing
    if args.is_testing:
        output_dir = os.path.join(args.res_dir, args.train_data+"2"+ args.test_data)
        if rank == 0:
            print(f"output_dir: {output_dir}")
        test(checkpoint_path, dataloader_val, model, device, output_dir, args)
        num_param = count_parameters(model)
        if rank == 0:
            print('-------------------------------------------------------')
            print('DexiNed # of Parameters:')
            print(num_param)
            print('-------------------------------------------------------')
        return

    criterion = bdcn_loss2
    param_dicts = [
            {"params": [p for n, p in model.named_parameters() if ("upblocks" not in n) and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if ("upblocks" in n) and p.requires_grad],
                "lr": args.lr * 0.2,
            },
        ]
    optimizer = optim.Adam(param_dicts,
                            lr=args.lr,
                            weight_decay=args.wd)
    
    # Main training loop
    seed=1021
    adjust_lr = args.adjust_lr
    lr2= args.lr
    for epoch in range(ini_epoch,args.epochs):
        if rank == 0:
            print("Training Epoch :", epoch)
        if epoch%7==0:

            seed = seed+1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("------ Random seed applied-------------")
        # Create output directories
        if adjust_lr is not None:
            if epoch in adjust_lr:
                lr2 = lr2*0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr2

        output_dir_epoch = os.path.join(args.output_dir,args.train_data, str(epoch))
        img_test_dir = os.path.join(output_dir_epoch, args.test_data + '_res')
        os.makedirs(output_dir_epoch,exist_ok=True)
        os.makedirs(img_test_dir,exist_ok=True)
        avg_loss = train_one_epoch(rank, 
                        epoch,
                        dataloader_train,
                        model,
                        criterion,
                        optimizer,
                        device,
                        args.log_interval_vis,
                        tb_writer,
                        args=args)
        if rank==0:
            validate_one_epoch(epoch,
                            dataloader_val,
                            model,
                            device,
                            img_test_dir,
                            arg=args)
            # Save model after end of every epoch
            torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))
        if tb_writer is not None:
            tb_writer.add_scalar('loss',
                                avg_loss,
                                epoch+1)
        if rank == 0:               
            print('Current learning rate> ', optimizer.param_groups[0]['lr'])
    num_param = count_parameters(model)
    if rank == 0:
        print('-------------------------------------------------------')
        print('DexiNed, # of Parameters:')
        print(num_param)
        print('-------------------------------------------------------')

if __name__ == '__main__':
    
    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    args = parse_args()
    args.DDP = True
    args.nodes = 1
    args.GPU_ids = '1,3'
    args.gpus = len(args.GPU_ids.split(','))
    args.node_rank =  0
    args.world_size = args.nodes * args.gpus
    
    ## Print arguments
    # for key, value in vars(args).items():
    #     print(f"{key}: {value}")
    
    import socket
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))  # Bind to localhost with port 0 to find a free port
            return s.getsockname()[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['MASTER_PORT'] = str(find_free_port())
    mp.set_start_method('spawn')
    mp.spawn(main, nprocs=args.world_size, args=(args,), join=True)
