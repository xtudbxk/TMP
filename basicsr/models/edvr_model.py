import torch
from collections import OrderedDict
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class EDVRModel(VideoBaseModel):
    """EDVR Model.

    Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.  # noqa: E501
    """

    def __init__(self, opt):
        super(EDVRModel, self).__init__(opt)
        if self.is_train:
            self.train_tsa_iter = opt['train'].get('tsa_iter')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        dcn_lr_mul = train_opt.get('dcn_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for dcn with {dcn_lr_mul}.')
        if dcn_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate dcn params and normal params for different lr
            normal_params = []
            dcn_params = []
            for name, param in self.net_g.named_parameters():
                if 'dcn' in name:
                    dcn_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': dcn_params,
                    'lr': train_opt['optim_g']['lr'] * dcn_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.train_tsa_iter:
            if current_iter == 1:
                logger = get_root_logger()
                logger.info(f'Only train TSA module for {self.train_tsa_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'fusion' not in name:
                        param.requires_grad = False
            elif current_iter == self.train_tsa_iter:
                logger = get_root_logger()
                logger.warning('Train all the parameters.')
                for param in self.net_g.parameters():
                    param.requires_grad = True

        super(EDVRModel, self).optimize_parameters(current_iter)

    def _tb_display(self, tb_logger, step):
        _b, _t, _c, _h, _w = self.lq.shape
        self.lq.requires_grad = True
        self.output = self.net_g(self.lq)

        # for attentions among pixels 
        loss = torch.sum(torch.abs(self.output[:, :, _h*4//2, _w*4//2] - self.gt[:, :, _h*4//2, _w*4//2]))
        loss.backward()
        g = self.lq.grad

        # compute the metrics
        m = torch.mean(torch.mean(torch.abs(g), dim=2), dim=0)

        h_len = int(_h*0.1)
        w_len = int(_w*0.1)

        m_sum = torch.sum(torch.sum(m, axis=1), axis=1)
        m_inner = torch.sum(torch.sum(m[:, _h//2-h_len//2:_h//2+h_len//2, _w//2-w_len//2:_w//2+w_len//2], axis=1), axis=1)
        m_outter = m_sum - m_inner
        tmp = m_outter / m_inner[_t//2]

        tb_logger.add_scalar(f'display/lossgrads/temporal', (torch.sum(m_sum/m_sum[_t//2])-1)/(_t-1) , step)
        tb_logger.add_scalar(f'display/lossgrads/spatial', torch.sum(tmp)-tmp[_t//2] , step)

        self.lq.requires_grad = False
