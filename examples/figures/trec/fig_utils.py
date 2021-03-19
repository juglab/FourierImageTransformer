from matplotlib import pyplot as plt

from matplotlib import gridspec

from fit.utils.utils import denormalize, PSNR
import numpy as np

def figure(name, sinos, fbp_imgs, img_pred_fc, pred_img, imgs, mean, std, circle):
    fig = plt.figure(figsize=(14, 3*8 + 0.3) )
    gs = gridspec.GridSpec(9, 5, width_ratios=[2,3,3,3,3], height_ratios=[.3, 3, 3, 3, 3,3,3,3,3]) 
    t0 = plt.subplot(gs[0,0])
    t1 = plt.subplot(gs[0,1])
    t2 = plt.subplot(gs[0,2])
    t3 = plt.subplot(gs[0,3])
    t4 = plt.subplot(gs[0,4])

    ax0 = [plt.subplot(gs[1,i]) for i in range(5)]
    ax1 = [plt.subplot(gs[2,i]) for i in range(5)]
    ax2 = [plt.subplot(gs[3,i]) for i in range(5)]
    ax3 = [plt.subplot(gs[4,i]) for i in range(5)]
    ax4 = [plt.subplot(gs[5,i]) for i in range(5)]
    ax5 = [plt.subplot(gs[6,i]) for i in range(5)]
    ax6 = [plt.subplot(gs[7,i]) for i in range(5)]
    ax7 = [plt.subplot(gs[8,i]) for i in range(5)]

    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0.02, wspace = 0.02)

    t0.xaxis.set_major_locator(plt.NullLocator())
    t0.yaxis.set_major_locator(plt.NullLocator())
    t0.axis('off');
    t0.text(0.5, 0.5, 'Sinogram', fontsize=24, va='center', ha='center');

    t1.xaxis.set_major_locator(plt.NullLocator())
    t1.yaxis.set_major_locator(plt.NullLocator())
    t1.axis('off');
    t1.text(0.5, 0.5, 'FBP', fontsize=24, va='center', ha='center');

    t2.xaxis.set_major_locator(plt.NullLocator())
    t2.yaxis.set_major_locator(plt.NullLocator())
    t2.axis('off');
    t2.text(0.5, 0.5, 'iFFT of FC', fontsize=24, va='center', ha='center');

    t3.xaxis.set_major_locator(plt.NullLocator())
    t3.yaxis.set_major_locator(plt.NullLocator())
    t3.axis('off');
    t3.text(0.5, 0.5, 'Prediction', fontsize=24, va='center', ha='center');

    t4.xaxis.set_major_locator(plt.NullLocator())
    t4.yaxis.set_major_locator(plt.NullLocator())
    t4.axis('off');
    t4.text(0.5, 0.5, 'Ground Truth', fontsize=24, va='center', ha='center');

    def row(ax0, ax1, ax2, ax3, ax4, sino, fbp_img, img_pred_fc, pred_img, img):
        fbp_img = denormalize(fbp_img, mean, std)
        fbp_img *= circle
        
        img_pred_fc = denormalize(img_pred_fc, mean, std)
        img_pred_fc *= circle
        
        pred_img = denormalize(pred_img, mean, std)
        pred_img *= circle
        
        img = denormalize(img, mean, std)
        img *= circle
        
        ax0.xaxis.set_major_locator(plt.NullLocator())
        ax0.yaxis.set_major_locator(plt.NullLocator())
        ax0.imshow(sino.T, cmap='gray')

        ax1.xaxis.set_major_locator(plt.NullLocator())
        ax1.yaxis.set_major_locator(plt.NullLocator())
        ax1.imshow(fbp_img, cmap='gray')#, vmin=img.min(), vmax=img.max())

        ax2.xaxis.set_major_locator(plt.NullLocator())
        ax2.yaxis.set_major_locator(plt.NullLocator())
        ax2.imshow(img_pred_fc.detach().numpy(), cmap='gray')#, vmin=img.min(), vmax=img.max())

        ax3.xaxis.set_major_locator(plt.NullLocator())
        ax3.yaxis.set_major_locator(plt.NullLocator())
        ax3.imshow(pred_img.detach().numpy(), cmap='gray')#, vmin=img.min(), vmax=img.max())

        ax4.xaxis.set_major_locator(plt.NullLocator())
        ax4.yaxis.set_major_locator(plt.NullLocator())
        ax4.imshow(img, cmap='gray')#, vmin=img.min(), vmax=img.max())


    row(*ax0, sinos[0], fbp_imgs[0], img_pred_fc[0], pred_img[0], imgs[0])
    row(*ax1, sinos[1], fbp_imgs[1], img_pred_fc[1], pred_img[1], imgs[1])
    row(*ax2, sinos[2], fbp_imgs[2], img_pred_fc[2], pred_img[2], imgs[2])
    row(*ax3, sinos[3], fbp_imgs[3], img_pred_fc[3], pred_img[3], imgs[3])
    row(*ax4, sinos[4], fbp_imgs[4], img_pred_fc[4], pred_img[4], imgs[4])
    row(*ax5, sinos[5], fbp_imgs[5], img_pred_fc[5], pred_img[5], imgs[5])
    row(*ax6, sinos[6], fbp_imgs[6], img_pred_fc[6], pred_img[6], imgs[6])
    row(*ax7, sinos[7], fbp_imgs[7], img_pred_fc[7], pred_img[7], imgs[7])
    fig.savefig(name, bbox_inches='tight', pad_inches=0.0)
    
    
def figure_trecs(name, sinos, fbp_imgs, img_pred_fc, pred_img, imgs, mean, std, circle, x,y):
    fig = plt.figure(figsize=(13.7, 3*8 + 0.3) )
    gs = gridspec.GridSpec(9, 5, width_ratios=[1.7,3,3,3,3], height_ratios=[.3, 3, 3, 3, 3,3,3,3,3]) 
    t0 = plt.subplot(gs[0,0])
    t1 = plt.subplot(gs[0,1])
    t2 = plt.subplot(gs[0,2])
    t3 = plt.subplot(gs[0,3])
    t4 = plt.subplot(gs[0,4])

    ax0 = [plt.subplot(gs[1,i]) for i in range(5)]
    ax1 = [plt.subplot(gs[2,i]) for i in range(5)]
    ax2 = [plt.subplot(gs[3,i]) for i in range(5)]
    ax3 = [plt.subplot(gs[4,i]) for i in range(5)]
    ax4 = [plt.subplot(gs[5,i]) for i in range(5)]
    ax5 = [plt.subplot(gs[6,i]) for i in range(5)]
    ax6 = [plt.subplot(gs[7,i]) for i in range(5)]
    ax7 = [plt.subplot(gs[8,i]) for i in range(5)]

    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0.02, wspace = 0.02)

    t0.xaxis.set_major_locator(plt.NullLocator())
    t0.yaxis.set_major_locator(plt.NullLocator())
    t0.axis('off');
#     t0.text(0.5, 0.5, 'Sinogram', fontsize=24, va='center', ha='center');

    t1.xaxis.set_major_locator(plt.NullLocator())
    t1.yaxis.set_major_locator(plt.NullLocator())
    t1.axis('off');
    t1.text(0.5, 0.5, 'FBP Baseline', fontsize=24, va='center', ha='center');

    t2.xaxis.set_major_locator(plt.NullLocator())
    t2.yaxis.set_major_locator(plt.NullLocator())
    t2.axis('off');
    t2.text(0.5, 0.5, 'TRec (Ours)', fontsize=24, va='center', ha='center');

    t3.xaxis.set_major_locator(plt.NullLocator())
    t3.yaxis.set_major_locator(plt.NullLocator())
    t3.axis('off');
    t3.text(0.5, 0.5, 'TRec + FBP (Ours)', fontsize=24, va='center', ha='center');

    t4.xaxis.set_major_locator(plt.NullLocator())
    t4.yaxis.set_major_locator(plt.NullLocator())
    t4.axis('off');
    t4.text(0.5, 0.5, 'Ground Truth', fontsize=24, va='center', ha='center');

    def row(ax0, ax1, ax2, ax3, ax4, sino, fbp_img, img_pred_fc, pred_img, img):
        fbp_img = denormalize(fbp_img, mean, std)
        fbp_img *= circle
        
        img_pred_fc = denormalize(img_pred_fc, mean, std)
        img_pred_fc *= circle
        
        pred_img = denormalize(pred_img, mean, std)
        pred_img *= circle
        
        img = denormalize(img, mean, std)
        img *= circle
        
        psnr_fbp = PSNR(fbp_img, img, img.max() - img.min())
        psnr_pred_fc = PSNR(img_pred_fc, img, img.max() - img.min())
        psnr_pred_img = PSNR(pred_img, img, img.max() - img.min())
        
        ax0.xaxis.set_major_locator(plt.NullLocator())
        ax0.yaxis.set_major_locator(plt.NullLocator())
        ax0.imshow(sino.T, cmap='gray')

        ax1.xaxis.set_major_locator(plt.NullLocator())
        ax1.yaxis.set_major_locator(plt.NullLocator())
        ax1.imshow(fbp_img, cmap='gray')#, vmin=img.min(), vmax=img.max())
        ax1.text(x, y, np.round(psnr_fbp.item(), 2), c='white') 

        ax2.xaxis.set_major_locator(plt.NullLocator())
        ax2.yaxis.set_major_locator(plt.NullLocator())
        ax2.imshow(img_pred_fc.detach().numpy(), cmap='gray')#, vmin=img.min(), vmax=img.max())
        ax2.text(x, y, np.round(psnr_pred_fc.item(), 2), c='white') 


        ax3.xaxis.set_major_locator(plt.NullLocator())
        ax3.yaxis.set_major_locator(plt.NullLocator())
        ax3.imshow(pred_img.detach().numpy(), cmap='gray')#, vmin=img.min(), vmax=img.max())
        ax3.text(x, y, np.round(psnr_pred_img.item(), 2), c='white') 


        ax4.xaxis.set_major_locator(plt.NullLocator())
        ax4.yaxis.set_major_locator(plt.NullLocator())
        ax4.imshow(img, cmap='gray')#, vmin=img.min(), vmax=img.max())


    row(*ax0, sinos[0], fbp_imgs[0], img_pred_fc[0], pred_img[0], imgs[0])
    row(*ax1, sinos[1], fbp_imgs[1], img_pred_fc[1], pred_img[1], imgs[1])
    row(*ax2, sinos[2], fbp_imgs[2], img_pred_fc[2], pred_img[2], imgs[2])
    row(*ax3, sinos[3], fbp_imgs[3], img_pred_fc[3], pred_img[3], imgs[3])
    row(*ax4, sinos[4], fbp_imgs[4], img_pred_fc[4], pred_img[4], imgs[4])
    row(*ax5, sinos[5], fbp_imgs[5], img_pred_fc[5], pred_img[5], imgs[5])
    row(*ax6, sinos[6], fbp_imgs[6], img_pred_fc[6], pred_img[6], imgs[6])
    row(*ax7, sinos[7], fbp_imgs[7], img_pred_fc[7], pred_img[7], imgs[7])
    fig.savefig(name, bbox_inches='tight', pad_inches=0.0)