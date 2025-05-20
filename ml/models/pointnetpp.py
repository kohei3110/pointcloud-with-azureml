import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
    """
    Calculate Euclid distance between each point in src and dst.
    Input:
        src: source points, [B, C, N]
        dst: target points, [B, C, M]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, C, N = src.shape
    _, _, M = dst.shape

    # (B, N, C)
    src_t = src.transpose(1, 2)
    # (B, M, C)
    dst_t = dst.transpose(1, 2)

    # (B, N, M)
    dist = torch.sum(src_t[:, :, None, :] ** 2, dim=-1) + \
           torch.sum(dst_t[:, None, :, :] ** 2, dim=-1) - \
           2 * torch.matmul(src_t, dst_t.transpose(1, 2))

    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, C, N]
        idx: sample index data, [B, S, K]
    Return:
        new_points:, indexed points data, [B, C, S, K]
    """
    raw_shape = idx.shape
    B, C, N = points.shape
    idx = idx.reshape(B, -1)
    idx_expand = idx.unsqueeze(1).expand(-1, C, -1)
    
    # Convert to long for indexing
    idx_expand = idx_expand.long()
    
    # Ensure valid indices (clamp between 0 and N-1)
    idx_expand = torch.clamp(idx_expand, 0, N-1)
    
    points_t = points.transpose(1, 2)  # [B, N, C]
    result = torch.gather(points_t, 1, idx_expand.transpose(1, 2))  # [B, S*K, C]
    result = result.transpose(1, 2)  # [B, C, S*K]
    
    return result.reshape(B, C, *raw_shape[1:])

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: point cloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled point cloud index, [B, npoint]
    """
    device = xyz.device
    B, C, N = xyz.shape
    
    # Transpose for easier computation
    xyz_t = xyz.transpose(1, 2)  # [B, N, C]
    
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    
    # Randomly select the first point
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz_t[batch_indices, farthest, :].view(B, 1, C)
        
        # Calculate distance to the selected centroid
        dist = torch.sum((xyz_t - centroid) ** 2, -1)
        
        # Update distance if current distance is smaller
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # Find the farthest point
        farthest = torch.max(distance, -1)[1]
    
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, C, N]
        new_xyz: query points, [B, C, S]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, C, N = xyz.shape
    _, _, S = new_xyz.shape
    
    # Calculate pairwise distance between each query point and all points
    dist = square_distance(new_xyz, xyz)
    
    # Find points within radius
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, S, 1)
    dist_mask = dist > radius ** 2
    group_idx[dist_mask] = N
    
    # Sort and select the first nsample points
    group_idx, _ = torch.sort(group_idx, dim=-1)
    group_idx = group_idx[:, :, :nsample]  # Truncate to nsample
    
    # If we have less than nsample points, pad with the first point
    mask = group_idx == N
    if mask.any():
        # For each sample in batch and each query point, get the first valid index or 0
        first_valid_idx = torch.zeros(B, S, 1, dtype=torch.long).to(device)
        for b in range(B):
            for s in range(S):
                valid_idx = group_idx[b, s, group_idx[b, s] < N]
                if valid_idx.shape[0] > 0:
                    first_valid_idx[b, s, 0] = valid_idx[0]
        
        # Use broadcasting to fill masked positions with first valid indices
        group_idx = torch.where(mask, first_valid_idx.repeat(1, 1, nsample), group_idx)
    
    return group_idx

class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        # Create MLP layers
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points: sample points feature data, [B, D', S]
        """
        B, C, N = xyz.shape
        
        if self.group_all:
            # Use all points
            new_xyz = torch.zeros(B, C, 1).to(xyz.device)
            grouped_xyz = xyz.view(B, C, 1, N)
            if points is not None:
                grouped_points = points.view(B, -1, 1, N)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=1)
            else:
                grouped_points = grouped_xyz
        else:
            # Sample points
            new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
            
            # Group points
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            
            # Normalize
            grouped_xyz -= new_xyz.view(B, C, self.npoint, 1)
            
            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=1)
            else:
                grouped_points = grouped_xyz
        
        # Apply MLP
        for i, conv in enumerate(self.mlp_convs):
            grouped_points = F.relu(self.mlp_bns[i](conv(grouped_points)))
        
        # Max pooling
        new_points = torch.max(grouped_points, -1)[0]
        
        return new_xyz, new_points

class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: sampled points data, [B, D', S]
        Return:
            new_points: upsampled points data, [B, D'', N]
        """
        B, C, N = xyz1.shape
        _, _, S = xyz2.shape
        
        # If no sampled points, directly copy features
        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            # Calculate distance for interpolation
            dist = square_distance(xyz1, xyz2)
            dist, idx = torch.topk(dist, k=3, dim=-1, largest=False, sorted=True)
            
            # Normalize distances
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            # Interpolate features
            idx = idx.view(B, N * 3)
            interpolated_points = index_points(points2, idx).view(B, -1, N, 3)
            interpolated_points = torch.sum(interpolated_points * weight.view(B, 1, N, 3), dim=-1)
        
        # Concatenate with existing features and apply MLP
        if points1 is not None:
            new_points = torch.cat([interpolated_points, points1], dim=1)
        else:
            new_points = interpolated_points
        
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](conv(new_points)))
        
        return new_points

class PointNetPlusPlusSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNetPlusPlusSeg, self).__init__()
        
        # Encoder
        self.sa1 = SetAbstraction(512, 0.2, 32, 3, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = SetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)
        
        # Decoder
        self.fp3 = FeaturePropagation(1024 + 256, [256, 256])
        self.fp2 = FeaturePropagation(256 + 128, [256, 128])
        self.fp1 = FeaturePropagation(128, [128, 128, 128])
        
        # Classifier
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    
    def forward(self, x):
        """
        Input:
            x: point cloud data, [B, 3, N]
        Return:
            x: segmentation scores, [B, num_classes, N]
        """
        B, _, N = x.shape
        
        # Encoder
        l0_xyz = x
        l0_points = None
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        
        # Classifier
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()