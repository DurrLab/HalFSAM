import cv2
import torch
import torch.nn as nn
import numpy as np
from torchmetrics.classification import BinaryJaccardIndex
import ipdb
# from chamferdist import ChamferDistance

'''
Metrics of consistency loss.

    Triplet loss is implemented and is identical to Torch version.
    Quadruplet loss is implemented, no Torch version available.
    The IoU is not used, we used torchmetric library.
    Distance Transform uses cv2 distance transform in Numpy.

    Deprecated: chamfer distance
'''
from torchmetrics import Metric
# from torchmetrics.functional.classification import binary_precision_recall_curve
# from sklearn.metrics import average_precision_score, precision_recall_curve

class EdgeDetectionMetrics(Metric):
    def __init__(self, n_thresholds=100, **kwargs):
        super().__init__(**kwargs)
        self.n_thresholds = n_thresholds
        self.add_state("hist_counts", default=torch.zeros(3, n_thresholds, dtype=torch.float64), 
                       dist_reduce_fx="sum")
        self.add_state("ois_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_images", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("all_f1", default=[], dist_reduce_fx='cat')

        # Thresholds stay on GPU but use float32 for memory reduction
        self.register_buffer("thresholds", torch.linspace(0, 1, n_thresholds, dtype=torch.float64))

        # Debug
        # self.add_state("all_preds", default=[], dist_reduce_fx='cat')
        # self.add_state("all_targets", default=[], dist_reduce_fx='cat')
        # self.add_state("sk_ois_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Full batch processing on GPU with memory optimizations"""
        preds = preds.double()
        target = target.bool()
        # Debug 
        # # Store flattened predictions/targets for AP calculation
        # self.all_preds.append(preds.flatten())
        # self.all_targets.append(target.bool().flatten())

        # Process OIS on GPU
        # for img_pred, img_target in zip(preds.unbind(0), target.unbind(0)):
        #     precision, recall, thresh = binary_precision_recall_curve(
        #         img_pred.flatten(), 
        #         img_target.flatten(), thresholds=self.thresholds
        #     )
        #     f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        #     # Print first 10 thresholds, precision, recall, f1
        #     print("Thresholds:", self.thresholds[:10])
        #     print("Precision:", precision[:10])
        #     print("Recall:", recall[:10])
        #     print("F1:", f1[:10])

        #     # # Debug: Compute f1 score using sklearn
        #     # img_pred_ = img_pred.flatten().cpu().numpy()
        #     # img_target_ = img_target.flatten().cpu().numpy()
        #     # precision_, recall_, thresh = precision_recall_curve(img_target_, img_pred_)
        #     # f1_ = 2 * (precision_ * recall_) / (precision_ + recall_
        #     #                              + 1e-8)
        #     # self.sk_ois_sum += f1_.max()
        #     self.ois_sum += torch.amax(f1)
        #     print(f"OIS from torchmetrics: {torch.amax(f1)}")
        #     self.total_images += 1

        # Vectorized histogram calculation 
        with torch.no_grad():
            # Reshape for broadcasting [B, H*W, 1] vs [1, 1, T]
            expanded_preds = preds.view(preds.shape[0], -1, 1)
            thresholds = self.thresholds.view(1, 1, -1)
            
            # Boolean masks consume less memory than float
            above_thresh = (expanded_preds > thresholds)  # [B, H*W, T]
            target_flat = target.view(target.shape[0], -1, 1)  # [B, H*W, 1]
            
            # Calculate TP/FP/FN using boolean indexing
            tp = (above_thresh & target_flat).sum(dim=1)  # [B, T]
            fp = (above_thresh & ~target_flat).sum(dim=1)
            fn = (~above_thresh & target_flat).sum(dim=1)
            
            # Compute F1 score for each threshold
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            self.all_f1.append(f1)

            # # Print first 10 thresholds, precision, recall, f1
            # print("Thresholds:", self.thresholds[:10])
            # print("Precision:", precision[0, :10])
            # print("Recall:", recall[0, :10])
            # print("F1:", f1[0, :10])
            # # Print F1 max values for each image in batch
            # print(f"OIS our implementation: {f1.max(dim=1)}")

            # Compute OIS: maximum F1 score for each image, then average across images
            max_f1_per_image, _ = torch.max(f1, dim=1)  # [B]
            self.ois_sum += max_f1_per_image.sum()
            self.total_images += preds.shape[0]

            # # Aggregate across batch
            self.hist_counts += torch.stack([tp.sum(0), fp.sum(0), fn.sum(0)])

        # Explicit memory cleanup
        del expanded_preds, thresholds, above_thresh, target_flat, tp, fp, fn
        torch.cuda.empty_cache()

    def compute(self):
        """Final metric computation with reduced precision"""
        # # Convert to float32 for calculation if needed
        tp_total = self.hist_counts[0].double()
        fp_total = self.hist_counts[1].double()
        fn_total = self.hist_counts[2].double()

        precision = tp_total / (tp_total + fp_total + 1e-8)
        recall = tp_total / (tp_total + fn_total + 1e-8)
        # f1_ = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # print(f"Old ODS: {f1_.max()}")

        # Take average F1 across all thresholds
        print(len(self.all_f1))
        print(self.all_f1[0].shape)
        # f1_grid = torch.concatenate(self.all_f1, dim=0)
        f1_grid = torch.cat([f.to(self.device) for f in self.all_f1], dim=0)
        f1 = f1_grid.mean(dim=0)

        # # Print first 10 thresholds, precision, recall, f1
        # print("Thresholds:", self.thresholds[:10])
        # print("Precision Py:", precision[:10])
        # print("Recall Py:", recall[:10])
        # print("F1 Py:", f1[:10])

        # debug
        # # Concatenate all predictions/targets
        # preds = torch.cat(self.all_preds).cpu().numpy()
        # targets = torch.cat(self.all_targets).cpu().numpy()

        # # Compute official AP using sklearn (for validation)
        # sklearn_ap = average_precision_score(targets, preds)

        # # Compute precision, recall, f1 using sklearn over preds/targets
        # precision_, recall_, thresh = precision_recall_curve(targets, preds)
        # f1_ = 2 * (precision_ * recall_) / (precision_ + recall_
        #                                  + 1e-8)

        # # Print first 10 thresholds, precision, recall, f1 for sklearn
        # print("Thresholds SK:", thresh[:10])
        # print("Precision SK:", precision_[:10])
        # print("Recall SK:", recall_[:10])
        # print("F1 SK:", f1_[:10])

        return {
            'ODS': f1.max(),
            'OIS': self.ois_sum.float() / self.total_images,
            'AP': self._compute_ap(precision, recall),
            # 'AP_sklearn': sklearn_ap,
            # 'ODS_sklearn': f1_.max(),
            # 'OIS_sklearn': self.sk_ois_sum.float() / self.total_images
        }

    def _compute_ap(self, precision, recall):
        """Numerically stable AP calculation"""
        # Sort by recall and compute AUC
        recall, idx = torch.sort(recall)
        precision = precision[idx]
        
        # Pad boundaries with 0 and 1
        precision = torch.cat([torch.tensor([0.0], device=precision.device), precision])
        recall = torch.cat([torch.tensor([0.0], device=recall.device), recall])
        
        return torch.trapz(precision, recall)  # Trapezoidal integration

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = nn.functional.pairwise_distance(anchor,positive)
        distance_negative = nn.functional.pairwise_distance(anchor,negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class QuadrupletLoss(torch.nn.Module):
    def __init__(self, margin1=1.0, margin2=0.5):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, anchor, positive, negative1, negative2):

        squarred_distance_pos = nn.functional.pairwise_distance(anchor, positive)
        squarred_distance_neg = nn.functional.pairwise_distance(anchor, negative1)
        squarred_distance_neg_b = nn.functional.pairwise_distance(negative1, negative2)

        quadruplet_loss = torch.relu(squarred_distance_pos - squarred_distance_neg + self.margin1) + \
                          torch.relu(squarred_distance_pos - squarred_distance_neg_b + self.margin2)

        return quadruplet_loss.mean()


'''input must be binary image, input is (h,w,c)'''
def distanceTransform(img,isTensor=False):
    if isTensor:
        img = img.numpy()
    
    grey = img[0,:,:] # convert to grey scale image
    grey = grey.astype(np.uint8)

    # Apply distance transform using open-cv
    trans = cv2.distanceTransform(grey, distanceType=cv2.DIST_L2, maskSize=3, dstType=cv2.CV_8U)
    trans = cv2.normalize(trans, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if isTensor:
        trans = torch.from_numpy(trans)

    return trans

# def chamferDistance(source,target):
#     chamferdist = ChamferDistance()
#     dist_bidirectional = chamferdist(source/255, target/255, bidirectional=True)
#     return dist_bidirectional.detach().cpu().item()

def compute_IoU(preds, gt, thresh=0.5):
    """
    Compute IoU for batches of predictions and ground truth.
    
    Args:
    preds (torch.Tensor): Predictions tensor of shape [B, C, H, W] or [B, Time, C, H, W]
    gt (torch.Tensor): Ground truth tensor of shape [B, C, H, W] or [B, Time, C, H, W]
    thresh: threshold for binarization

    Returns:
    torch.Tensor: IoU scores for each class
    """
    # Ensure inputs are on the same device
    device = preds.device

    # Check if we need to apply sigmoid
    preds = torch.sigmoid(preds)

    # Binarize predictions & GT
    preds = (preds > thresh).int()
    gt = (gt>thresh).int()
    
    # Check if it's a 5D tensor
    if preds.dim() == 5:  # [B, Time, C, H, W]
        B, T, C, H, W = preds.shape
        preds = preds.reshape(B*T, C, H, W)
        gt = gt.reshape(B*T, C, H, W)
    
    # Ignore frames which don't have any gt edges
    ignore_mask = (gt.sum(dim=(1, 2, 3)) == 0)
    preds = preds[~ignore_mask]
    gt = gt[~ignore_mask]

    # Initialize BinaryJaccardIndex
    jaccard = BinaryJaccardIndex(threshold=thresh).to(device)

    # Compute iou
    iou = jaccard(preds, gt)
    return iou

def compute_temporal_consistency_batch(
    edge_maps: torch.Tensor,  # Input shape: (B, C, H, W)
    flow_maps: torch.Tensor,   # Optical flow shape: (B, T, 2, H, W)
    edgeBinaryThreshold=0.2, # Use 0 for binary labels (gt)
    distTransfomThreshold=250,
    boundary=25,
    black_border=55
) -> float:
    """
    Batch-processed temporal consistency for edge map sequences
    From frames index 1 to T-1, calculate the temporal consistency score
    Args:
        edge_maps: Batch of edge map sequences (B, T, C, H, W) or (B, C, H, W)
        flow_maps: Corresponding optical flow maps (B, T, 2, H, W) or (B, 2, H, W)
        edgeBinaryThreshold: Edge binarization threshold
        distTransfomThreshold: Distance transform threshold
    
    Returns:
        Average temporal consistency score across all batches and frames
    """
    # Convert tensors to numpy arrays
    if edge_maps.dim() == 4:  # (T, C, H, W)
        edge_maps = edge_maps.unsqueeze(0)  # Add batch dimension
    if flow_maps.dim() == 4:  # (B, T, 2, H, W)
        flow_maps = flow_maps.unsqueeze(0)  # Add batch dimension

    edges_np = edge_maps.cpu().numpy()  # (B, T, H, W)
    flows_np = flow_maps.cpu().numpy()             # (B, T, 2, H, W)
    
    B, T, C, H, W = edges_np.shape
    total_score = 0.0
    valid_pairs = 0

    debug_dir = 'debug'

    # # Create border mask once per batch
    # border_mask = np.ones((H, W), dtype=bool)
    # border_mask[:black_border+boundary, :] = False
    # border_mask[-(black_border+boundary):, :] = False
    # border_mask[:, :boundary] = False
    # border_mask[:, -boundary:] = False

    for b in range(B):
        for t in range(1, T):
            try:
                # Get consecutive frames and flow
                curr_edge = edges_np[b, t].squeeze()
                prev_edge = edges_np[b, t-1].squeeze()
                flow = flows_np[b, t]

                # # Apply border mask
                # curr_edge = curr_edge * border_mask
                # prev_edge = prev_edge * border_mask

                # Process flow map (same as original)
                flow_map = flow.transpose(1, 2, 0)  # (H, W, 2)
                invalid_mask = (flow_map[:, :, 0] == -20) & (flow_map[:, :, 1] == -20)
                flow_map[invalid_mask] = 0

                # Warp current edge using optical flow
                grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
                map_x = (grid_x + flow_map[:, :, 1]).astype(np.float32)
                map_y = (grid_y + flow_map[:, :, 0]).astype(np.float32)
                
                map_x = np.clip(map_x, 0, W-1)
                map_y = np.clip(map_y, 0, H-1)
                
                warped_edge = cv2.remap(curr_edge, map_x, map_y, cv2.INTER_LINEAR)

                # Write warped_edge to disk
                # cv2.imwrite(f"{debug_dir}/warped_edge_{b}_{t}.png", (warped_edge * 255).astype(np.uint8))

                # Distance transform processing
                # Maybe replace this with edge dilation function
                def process_edge(edge):
                    _, binary = cv2.threshold(edge, edgeBinaryThreshold, 255, cv2.THRESH_BINARY)
                    dt = cv2.distanceTransform((255 - binary).astype(np.uint8), cv2.DIST_L2, 3)
                    dt = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    _, dt = cv2.threshold(255 - dt, distTransfomThreshold, 255, cv2.THRESH_BINARY)
                    return dt / 255.0  # Normalize to 0-1

                warped_dt = process_edge(warped_edge)
                prev_dt = process_edge(prev_edge)
                
                # # Write warped_dt and prev_dt to disk
                # cv2.imwrite(f"{debug_dir}/warped_dt_{b}_{t}.png", (warped_dt * 255).astype(np.uint8))
                # cv2.imwrite(f"{debug_dir}/prev_dt_{b}_{t}.png", (prev_dt * 255).astype(np.uint8))

                # Calculate Jaccard Index
                intersection = np.logical_and(warped_dt, prev_dt).sum()
                union = np.logical_or(warped_dt, prev_dt).sum()
                
                if union > 0:
                    score = intersection / union
                    total_score += score
                    valid_pairs += 1

            except Exception as e:
                print(f"Error batch {b}, frame {t}: {str(e)}")
                continue

    return total_score / valid_pairs if valid_pairs > 0 else 0.0

if __name__ == '__main__':
    # anchor = torch.randn(2, 3, 128, 128, requires_grad=True)
    # positive = torch.randn(2, 3, 128, 128, requires_grad=True)
    # negative = torch.randn(2, 3, 128, 128, requires_grad=True)
    # trdneigb = torch.randn(2, 3, 128, 128, requires_grad=True)

    # #consistency = nn.TripletMarginLoss(margin=1)
    # #output = consistency(anchor, positive, negative)
    # #print('tested triplet loss with random inputs: ',output.item())

    # criterion = TripletLoss()
    # output = criterion(anchor, positive, negative)
    # print('tested Triplet loss with random inputs: ',output.item())

    # criterion = QuadrupletLoss()
    # output = criterion(anchor, positive, negative, trdneigb)
    # print('tested Quadplet loss with random inputs: ',output.item())
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create a 2D tensor with 
    preds = torch.tensor(
        [[0, 0.21, 0],
         [0.51, 0, 0.71],
         [0, 0.81, 0]]
    )
    gt = torch.tensor(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
    )

    preds = preds.unsqueeze(0).unsqueeze(0)
    gt = gt.unsqueeze(0).unsqueeze(0)

    # Initiallize edge detection metrics
    edge_detection_metrics = EdgeDetectionMetrics(n_thresholds=10)
    edge_detection_metrics.update(preds, gt)
    metrics = edge_detection_metrics.compute()
    print(metrics)

    # Compute the AUROC score using sklearn


    # # Create random 4D tensor [B, C, H, W]

    # B, C, H, W = 2, 1, 3, 3
    # preds_4d = torch.tensor([
    # [[[0, 1, 0],
    #   [1, 0, 1],
    #   [0, 1, 0]]],

    # [[[1, 0, 1],
    #   [0, 1, 0],
    #   [1, 0, 1]]]
    #                         ])
    # gt_4d = torch.tensor([
    # [[[0, 1, 0],
    #   [1, 0, 0],
    #   [0, 1, 0]]],

    # [[[0, 0, 0],
    #   [0, 0, 0],
    #   [0, 0, 0]]]
    #                         ])

    # # Create random 5D tensor [B, Time, C, H, W]
    # Time = 5
    # preds_5d = torch.randn(B, Time, C, H, W)
    # gt_5d = torch.rand(B, Time, C, H, W)

    # # Print 4D tensors
    # print("4D Tensor preds_4d:", preds_4d)
    # print("4D Tensor gt_4d:", gt_4d)

    # # Print 5D tensors
    # print("5D Tensor preds_5d:", preds_5d)
    # print("5D Tensor gt_5d:", gt_5d)

    # # Compute IoU for 4D tensor
    # iou_4d = compute_IoU(preds_4d, gt_4d)
    # print(f"IoU for 4D tensor: {iou_4d:.4f}")

    # # Compute IoU for 5D tensor
    # iou_5d = compute_IoU(preds_5d, gt_5d)
    # print(f"IoU for 5D tensor: {iou_5d:.4f}")
