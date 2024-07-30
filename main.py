import os
import sys
import pickle
import argparse
import time
from torch import maximum, optim
from torch.utils.tensorboard import SummaryWriter
import itertools
sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m_multimodal import DatasetH36M
from motion_pred.utils.dataset_humaneva_multimodal import DatasetHumanEva
from motion_pred.utils.visualization import render_animation
from models.motion_pred import *
from utils import util, valid_angle_check
from utils.metrics import *
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import random
import time

def recon_loss(Y_g, Y, Y_mm, Y_hg=None, Y_h=None):
    
    stat = torch.zeros(Y_g.shape[2])
    diff = Y_g - Y.unsqueeze(2) # TBMV
    dist = diff.pow(2).sum(dim=-1).sum(dim=0) # BM
    
    value, indices = dist.min(dim=1)

    loss_recon_1 = value.mean()
    
    diff = Y_hg - Y_h.unsqueeze(2) # TBMC
    loss_recon_2 = diff.pow(2).sum(dim=-1).sum(dim=0).mean()
    

    with torch.no_grad():
        ade = torch.norm(diff, dim=-1).mean(dim=0).min(dim=1)[0].mean()

    diff = Y_g[:, :, :, None, :] - Y_mm[:, :, None, :, :]
    
    mask = Y_mm.abs().sum(-1).sum(0) > 1e-6 

    dist = diff.pow(2) 
    with torch.no_grad():
        zeros = torch.zeros([dist.shape[1], dist.shape[2]], requires_grad=False).to(dist.device)# [b,m]    
        zeros.scatter_(dim=1, index=indices.unsqueeze(1).repeat(1, dist.shape[2]), src=zeros+dist.max()-dist.min()+1)
        zeros = zeros.unsqueeze(0).unsqueeze(3).unsqueeze(4)
        dist += zeros
    dist = dist.sum(dim=-1).sum(dim=0)

    value_2, indices_2 = dist.min(dim=1)
    loss_recon_multi = value_2[mask].mean()
    if torch.isnan(loss_recon_multi):
        loss_recon_multi = torch.zeros_like(loss_recon_1)
    
    mask = torch.tril(torch.ones([cfg.nk, cfg.nk], device=device)) == 0
    # TBMC
    
    yt = Y_g.reshape([-1, cfg.nk, Y_g.shape[3]]).contiguous()
    pdist = torch.cdist(yt, yt, p=1)[:, mask]
    return loss_recon_1, loss_recon_2, loss_recon_multi, ade, stat, (-pdist / 100).exp().mean()


def angle_loss(y):
    ang_names = list(valid_ang.keys())
    y = y.reshape([-1, y.shape[-1]])
    ang_cos = valid_angle_check.h36m_valid_angle_check_torch(
        y) if cfg.dataset == 'h36m' else valid_angle_check.humaneva_valid_angle_check_torch(y)
    loss = tensor(0, dtype=dtype, device=device)
    b = 1
    for an in ang_names:
        lower_bound = valid_ang[an][0]
        if lower_bound >= -0.98:
            # loss += torch.exp(-b * (ang_cos[an] - lower_bound)).mean()
            if torch.any(ang_cos[an] < lower_bound):
                # loss += b * torch.exp(-(ang_cos[an][ang_cos[an] < lower_bound] - lower_bound)).mean()
                loss += (ang_cos[an][ang_cos[an] < lower_bound] - lower_bound).pow(2).mean()
        upper_bound = valid_ang[an][1]
        if upper_bound <= 0.98:
            # loss += torch.exp(b * (ang_cos[an] - upper_bound)).mean()
            if torch.any(ang_cos[an] > upper_bound):
                # loss += b * torch.exp(ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).mean()
                loss += (ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).pow(2).mean()
    return loss


def loss_function(traj_est, traj, traj_multimodal, prior_lkh, prior_logdetjac, _lambda):
    lambdas = cfg.lambdas 
    nj = dataset.traj_dim // 3  #nj是关节数量

    Y_g = traj_est.view(traj_est.shape[0], traj.shape[1], traj_est.shape[1]//traj.shape[1], -1)[t_his:] # T B M V
    Y = traj[t_his:]
    Y_multimodal = traj_multimodal[t_his:]
    Y_hg=traj_est.view(traj_est.shape[0], traj.shape[1], traj_est.shape[1]//traj.shape[1], -1)[:t_his]
    Y_h= traj[:t_his]
    RECON, RECON_2, RECON_mm, ade, stat, JL = recon_loss(Y_g, Y, Y_multimodal,Y_hg, Y_h)
    # maintain limb length
    parent = dataset.skeleton.parents()
    tmp = traj[0].reshape([cfg.batch_size, nj, 3])
    pgt = torch.zeros([cfg.batch_size, nj + 1, 3], dtype=dtype, device=device)
    pgt[:, 1:] = tmp
    limbgt = torch.norm(pgt[:, 1:] - pgt[:, parent[1:]], dim=2)[None, :, None, :]
    tmp = traj_est.reshape([-1, cfg.batch_size, cfg.nk, nj, 3])
    pest = torch.zeros([tmp.shape[0], cfg.batch_size, cfg.nk, nj + 1, 3], dtype=dtype, device=device)
    pest[:, :, :, 1:] = tmp
    limbest = torch.norm(pest[:, :, :, 1:] - pest[:, :, :, parent[1:]], dim=4)#没懂这个是干啥的
    loss_limb = torch.mean((limbgt - limbest).pow(2).sum(dim=3))

    # angle loss
    loss_ang = angle_loss(Y_g)
    if _lambda < 0.1:
        _lambda *= 10
    else:
        _lambda = 1
    
    loss_r =  loss_limb * lambdas[1] + JL * lambdas[3] * _lambda  + RECON * lambdas[4] + RECON_mm * lambdas[5] \
            - prior_lkh.mean() * lambdas[6] + RECON_2 * lambdas[7]# - prior_logdetjac.mean() * lambdas[7]
    if loss_ang > 0:
        loss_r += loss_ang * lambdas[8]
    return loss_r, np.array([loss_r.item(), loss_limb.item(), loss_ang.item(),
                            JL.item(), RECON.item(), RECON_2.item(), RECON_mm.item(), ade.item(),
                            prior_lkh.mean().item(), prior_logdetjac.mean().item()]), stat#, indices_key, indices_2_key
   

def dct_transform_torch(data, dct_m, dct_n):
    '''
    B, 60, 35
    '''
    batch_size, features, seq_len = data.shape

    data = data.contiguous().view(-1, seq_len)  # [180077*60， 35]
    data = data.permute(1, 0)  # [35, b*60]

    out_data = torch.matmul(dct_m[:dct_n, :], data)  # [dct_n, 180077*60]
    out_data = out_data.permute(1, 0).contiguous().view(-1, features, dct_n)  # [b, 60, dct_n]
    return out_data

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

def train(epoch, stats):

    dct_m, i_dct_m = get_dct_matrix(cfg.t_his+cfg.t_pred)
    dct_m = torch.from_numpy(dct_m).float().to(device)
    i_dct_m = torch.from_numpy(i_dct_m).float().to(device)

    model.train()
    t_s = time.time()
    train_losses = 0
    train_grad = 0
    total_num_sample = 0
    n_modality = 10
    loss_names = ['LOSS', 'loss_limb', 'loss_ang', 'loss_DIV',
                  'RECON', 'RECON_2', 'RECON_multi', "ADE", 'p(z)', 'logdet']
    generator = dataset.sampling_generator(num_samples=cfg.num_data_sample, batch_size=cfg.batch_size,
                                           n_modality=n_modality)
    prior = torch.distributions.Normal(torch.tensor(0, dtype=dtype, device=device),
                                       torch.tensor(1, dtype=dtype, device=device))
    
    for traj_np, traj_multimodal_np in tqdm(generator):
        with torch.no_grad():

            bs, _, nj, _ = traj_np[..., 1:, :].shape # [bs, t_full, numJoints, 3]
            traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1) # bs, T, NumJoints*3
            traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous() # T, bs, NumJoints*3
            
            traj_multimodal_np = traj_multimodal_np[..., 1:, :]  # [bs, n_modality, t_full, NumJoints, 3]
            traj_multimodal_np = traj_multimodal_np.reshape([bs, n_modality, t_his + t_pred, -1]).transpose(
                [2, 0, 1, 3]) # [t_full, bs, n_modality, NumJoints*3]
            traj_multimodal = tensor(traj_multimodal_np, device=device, dtype=dtype)  # .permute(0, 2, 1).contiguous()
        
            X = traj[:t_his]
            Y = traj[t_his:]

        pred, a, b = model(traj)
        
        pred_tmp1 = pred.reshape([-1, pred.shape[-1] // 3, 3])
        pred_tmp = torch.zeros_like(pred_tmp1[:, :1, :])
        pred_tmp1 = torch.cat([pred_tmp, pred_tmp1], dim=1)
        pred_tmp1 = util.absolute2relative_torch(pred_tmp1, parents=dataset.skeleton.parents()).reshape(
            [-1, pred.shape[-1]])
        z, prior_logdetjac = pose_prior(pred_tmp1)
        prior_lkh = prior.log_prob(z).sum(dim=-1)
        
        loss, losses, stat = loss_function(pred.unsqueeze(2), traj, traj_multimodal, prior_lkh, prior_logdetjac, epoch / cfg.num_epoch)
        
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=100)
        train_grad += grad_norm
        optimizer.step()
        train_losses += losses
        
        total_num_sample += 1
        del loss
        
    scheduler.step()
    train_losses /= total_num_sample
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    lr = optimizer.param_groups[0]['lr']
    # average cost of log time 20s
    tb_logger.add_scalar('train_grad', train_grad / total_num_sample, epoch)
    
    logger.info('====> Epoch: {} Time: {:.2f} {}  lr: {:.5f} branch_stats: {}'.format(epoch, time.time() - t_s, losses_str , lr, stats))
    
    return stats

def get_multimodal_gt(dataset_test):
    all_data = []
    data_gen = dataset_test.iter_generator(step=cfg.t_his)
    for data, _ in tqdm(data_gen):
        # print(data.shape)
        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, t_his:, :])
        num_mult.append(len(ind[0]))
   
    num_mult = np.array(num_mult)
    logger.info('')
    logger.info('')
    logger.info('=' * 80)
    logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{pd.shape[0]}')
    logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{pd.shape[0]}')
    return traj_gt_arr

def get_prediction(data, model, sample_num, num_seeds=1, concat_hist=True):
    # 1 * total_len * num_key * 3
    dct_m, i_dct_m = get_dct_matrix(cfg.t_his+cfg.t_pred)
    dct_m = torch.from_numpy(dct_m).float().to(device)
    i_dct_m = torch.from_numpy(i_dct_m).float().to(device)
    traj_np = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    # 1 * total_len * ((num_key-1)*3)
    traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
    # total_len * 1 * ((num_key-1)*3)
    X = traj[:t_his]
    Y_gt = traj[t_his:]
    X = X.repeat((1, sample_num * num_seeds, 1))
    Y_gt = Y_gt.repeat((1, sample_num * num_seeds, 1))
    
    
    Y, mu, logvar = model(X)
    
    if concat_hist:
        
        X = X.unsqueeze(2).repeat(1, sample_num * num_seeds, cfg.nk, 1)
        Y = Y[t_his:].unsqueeze(1)
        Y = torch.cat((X, Y), dim=0)
    # total_len * batch_size * feature_size
    Y = Y.squeeze(1).permute(1, 0, 2).contiguous().cpu().numpy()
    # batch_size * total_len * feature_size
    if Y.shape[0] > 1:
        Y = Y.reshape(-1, cfg.nk * sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
    return Y


def test(model, epoch):
    stats_func = {'Diversity': compute_diversity, 'AMSE': compute_amse, 'FMSE': compute_fmse, 'ADE': compute_ade,
                  'FDE': compute_fde, 'MMADE': compute_mmade, 'MMFDE': compute_mmfde, 'MPJPE': mpjpe_error}
    stats_names = list(stats_func.keys())
    stats_names.extend(['ADE_stat', 'FDE_stat', 'MMADE_stat', 'MMFDE_stat', 'MPJPE_stat'])
    stats_meter = {x: AverageMeter() for x in stats_names}

    data_gen = dataset_test.iter_generator(step=cfg.t_his)
    num_samples = 0
    num_seeds = 1
    
    for i, (data, _) in tqdm(enumerate(data_gen)):
        if args.mode == 'train' and (i >= 500 and (epoch + 1) % 50 != 0 and (epoch + 1) < cfg.num_epoch - 100):
            break
        num_samples += 1
        gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)[:, t_his:, :]
        gt_multi = traj_gt_arr[i]
        if gt_multi.shape[0] == 1:
            continue
    
        pred = get_prediction(data, model, sample_num=1, num_seeds=num_seeds, concat_hist=False)
        pred = pred[:,:,t_his:,:]
        for stats in stats_names[:8]:
            val = 0
            branches = 0
            for pred_i in pred:
                # sample_num * total_len * ((num_key-1)*3), 1 * total_len * ((num_key-1)*3)
                v = stats_func[stats](pred_i, gt, gt_multi)
                val += v[0] / num_seeds
                if stats_func[stats](pred_i, gt, gt_multi)[1] is not None:
                    branches += v[1] / num_seeds
            stats_meter[stats].update(val)
            if type(branches) is not int:
                stats_meter[stats + '_stat'].update(branches)

    logger.info('=' * 80)
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + f'{stats_meter[stats].avg}'
        logger.info(str_stats)
    logger.info('=' * 80)


def visualize():
    def denomarlize(*data):
        out = []
        for x in data:
            x = x * dataset.std + dataset.mean
            out.append(x)
        return out

    def post_process(pred, data):
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
        if cfg.normalize_data:
            pred = denomarlize(pred)
        pred = np.concatenate((np.tile(data[..., :1, :], (pred.shape[0], 1, 1, 1)), pred), axis=2)
        pred[..., :1, :] = 0
        return pred

    def pose_generator():

        while True:
            data, data_multimodal, action = dataset_test.sample(n_modality=10)
            gt = data[0].copy()
            gt[:, :1, :] = 0

            poses = {'action': action, 'context': gt, 'gt': gt}
            with torch.no_grad():
                pred = get_prediction(data, model, 1)[0]
                pred = post_process(pred, data)
                for i in range(pred.shape[0]):
                    poses[f'{i}'] = pred[i]

            yield poses

    pose_gen = pose_generator()
    for i in tqdm(range(args.n_viz)):
        render_animation(dataset.skeleton, pose_gen, cfg.t_his, ncol=12, output='./results/{}/results/'.format(args.cfg), index_i=i)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        default='h36m')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--n_pre', type=int, default=8)
    parser.add_argument('--n_his', type=int, default=5)
    parser.add_argument('--n_viz', type=int, default=100)
    parser.add_argument('--num_coupling_layer', type=int, default=4)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    
    cfg = Config(f'{args.cfg}', test=args.test)
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    cfg.n_his = args.n_his
    if 'n_pre' not in cfg.specs.keys():
        cfg.n_pre = args.n_pre
    else:
        cfg.n_pre = cfg.specs['n_pre']
    cfg.num_coupling_layer = args.num_coupling_layer
    # cfg.nz = args.nz
    """data"""
    if 'actions' in cfg.specs.keys():
        act = cfg.specs['actions']
    else:
        act = 'all'
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls('train', t_his, t_pred, actions=act, use_vel=cfg.use_vel,
                          multimodal_path=cfg.specs[
                              'multimodal_path'] if 'multimodal_path' in cfg.specs.keys() else None,
                          data_candi_path=cfg.specs[
                              'data_candi_path'] if 'data_candi_path' in cfg.specs.keys() else None)
    dataset_test = dataset_cls('test', t_his, t_pred, actions=act, use_vel=cfg.use_vel,
                               multimodal_path=cfg.specs[
                                   'multimodal_path'] if 'multimodal_path' in cfg.specs.keys() else None,
                               data_candi_path=cfg.specs[
                                   'data_candi_path'] if 'data_candi_path' in cfg.specs.keys() else None)
    if cfg.normalize_data:
        dataset.normalize_data()
        dataset_test.normalize_data(dataset.mean, dataset.std)
    traj_gt_arr = get_multimodal_gt(dataset_test)
    """model"""
    model, pose_prior = get_model(cfg, dataset, cfg.dataset)
    
    model.float()
    pose_prior.float()
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_epoch_fix, nepoch=cfg.num_epoch)


    cp_path = 'results/h36m_nf/models/0025.p' if cfg.dataset == 'h36m' else 'results/humaneva_nf/models/0025.p'
    print('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    pose_prior.load_state_dict(model_cp['model_dict'])
    pose_prior.to(device)

    valid_ang = pickle.load(open('./data/h36m_valid_angle.p', "rb")) if cfg.dataset == 'h36m' else pickle.load(
        open('./data/humaneva_valid_angle.p', "rb"))
    if args.iter > 0:
        cp_path = cfg.model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])
        print("load done")
        
    if mode == 'train':
        model.to(device)
        overall_iter = 0
        stats = torch.zeros(cfg.nk)
        model.train()

        for i in range(args.iter, cfg.num_epoch):
            stats = train(i, stats) 
            if cfg.save_model_interval > 0 and (i + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test(model, i) 
                model.train()
                with to_cpu(model):
                    cp_path = cfg.model_path % (i + 1)
                    model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                    
                    pickle.dump(model_cp, open(cp_path, 'wb'))
                       
    elif mode == 'test':
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            test(model,args.iter) 

    
    elif mode == 'viz':
        model.to(device)
        model.eval()
        with torch.no_grad():
            visualize()
