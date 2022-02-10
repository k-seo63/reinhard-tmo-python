import argparse
import time
import cv2
import numpy as np
from scipy.stats.mstats import gmean


class ReinhardTMO:

    def __init__(self, args):
        self.img_fn = args.img
        self.img_hdr = self.read_img()
        self.lum = 0.2126 * self.img_hdr[:,:,2] + 0.7152 * self.img_hdr[:,:,1] + 0.0722 * self.img_hdr[:,:,0]

        self.alpha = 0.18   # 0.045, 0.09, 0.18, 0.36, 0.72
        self.lum_scaled = self.lum_scaling()
        self.lum_max = self.lum.max()

    def read_img(self):
        return cv2.imread(self.img_fn, cv2.IMREAD_ANYDEPTH)

    def lum_scaling(self):
        gmean_lum = gmean(self.lum.flatten() + 1e-6)
        lum_scaled = self.alpha / gmean_lum * self.lum
        return lum_scaled

    def change_luminance(self, lum_ldr):
        img_out = np.zeros(self.img_hdr.shape)
        for ch in range(3):
            img_out[:,:,ch] = self.img_hdr[:,:,ch] / self.lum * lum_ldr
        return img_out

    def gamma_correction(self, img, gamma=2.2):
        img_gamma = img**(1/gamma)
        return img_gamma

    def tmo_global(self):
        lum_ldr = (self.lum_scaled * (1 + self.lum_scaled / self.lum_max**2)) / (1 + self.lum_scaled)
        img_ldr = self.change_luminance(lum_ldr)
        img_ldr_gamma = self.gamma_correction(img_ldr)
        return np.clip(img_ldr_gamma * 255, 0, 255).astype('uint8')

    def tmo_local(self):
        scale = 8
        phi = 8
        lum_filtered = self.gaussian_filter()

        return 0

    def gaussian_filter(self):
        return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    # parser.add_argument('--alpha', type=float, required=False)
    args = parser.parse_args()

    reinhard = ReinhardTMO(args)

    start = time.time()
    img_tm = reinhard.tmo_global()
    process_time = time.time() - start
    print('process time = %f' % process_time)

    cv2.imshow("image", img_tm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
