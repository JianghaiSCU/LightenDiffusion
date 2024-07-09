import torch
import numpy as np
import utils
import os
import time
import torch.nn.functional as F


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=False)
            self.diffusion.model.eval()
        else:
            print('Pre-trained model path is missing!')

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):

                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                b, c, h, w = x_cond.shape
                img_h_64 = int(64 * np.ceil(h / 64.0))
                img_w_64 = int(64 * np.ceil(w / 64.0))
                x_cond = F.pad(x_cond, (0, img_w_64 - w, 0, img_h_64 - h), 'reflect')

                t1 = time.time()
                pred_x = self.diffusion.model(torch.cat((x_cond, x_cond),
                                                        dim=1))["pred_x"][:, :, :h, :w]
                t2 = time.time()

                utils.logging.save_image(pred_x, os.path.join(image_folder, f"{y[0]}"))
                print(f"processing image {y[0]}, time={t2 - t1}")



