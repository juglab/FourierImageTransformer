import argparse
import glob
import json
from os import mkdir
from os.path import exists

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from fit.datamodules.tomo_rec import MNISTTomoFourierTargetDataModule
from fit.modules import TRecTransformerModule
from fit.utils.tomo_utils import get_proj_coords, get_img_coords


def main():
    seed_everything(28122020)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_config")

    args = parser.parse_args()

    with open(args.exp_config) as f:
        conf = json.load(f)

    dm = MNISTTomoFourierTargetDataModule(root_dir=conf['root_dir'], batch_size=conf['batch_size'],
                                          num_angles=conf['num_angles'])
    dm.setup()

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()

    det_len = dm.gt_ds.get_ray_trafo().geometry.detector.shape[0]

    proj_xcoords, proj_ycoords, src_flatten = get_proj_coords(angles=dm.gt_ds.get_ray_trafo().geometry.angles,
                                                              det_len=det_len)
    target_xcoords, target_ycoords, dst_flatten, order = get_img_coords(img_shape=dm.IMG_SHAPE, det_len=det_len)

    model = TRecTransformerModule(d_model=256, y_coords_proj=proj_ycoords, x_coords_proj=proj_xcoords,
                                  y_coords_img=target_ycoords, x_coords_img=target_xcoords,
                                  src_flatten_coords=src_flatten, dst_flatten_coords=dst_flatten,
                                  dst_order=order,
                                  angles=dm.gt_ds.get_ray_trafo().geometry.angles, img_shape=dm.IMG_SHAPE,
                                  detector_len=det_len,
                                  init_bin_factor=3, bin_factor_cd=10, alpha=1.5,
                                  lr=0.0001, weight_decay=0.01, loss_switch=0.1, attention_type='linear', n_layers=8,
                                  n_heads=8, d_query=256 // 8, dropout=0.1, attention_dropout=0.1)

    if exists('lightning_logs'):
        print('Some experiments already exist. Abort.')
        return 0

    trainer = Trainer(max_epochs=conf['max_epochs'],
                      gpus=1,
                      checkpoint_callback=ModelCheckpoint(
                          filepath=None,
                          save_top_k=1,
                          verbose=False,
                          save_last=True,
                          monitor='Train/avg_val_mse',
                          mode='min',
                          prefix='best_val_loss_'
                      ),
                      deterministic=True)

    trainer.fit(model, train_dl, val_dl);

    model = TRecTransformerModule.load_from_checkpoint('lightning_logs/version_0/checkpoints/best_val_loss_-last.ckpt',
                                                       y_coords_proj=model.y_coords_proj,
                                                       x_coords_proj=model.x_coords_proj,
                                                       y_coords_img=model.y_coords_img,
                                                       x_coords_img=model.x_coords_img,
                                                       angles=model.angles)

    test_res = trainer.test(model, test_dl)[0]
    out_res = {
        "Mean PSNR": test_res["Mean PSNR"].item(),
        "SEM PSNR": test_res["SEM PSNR"].item()
    }
    with open('last_ckpt_results.json', 'w') as f:
        json.dump(out_res, f)

    best_path = glob.glob('lightning_logs/version_0/checkpoints/best_val_loss_-epoch*')[0]
    model = TRecTransformerModule.load_from_checkpoint(best_path,
        y_coords_proj=model.y_coords_proj,
        x_coords_proj=model.x_coords_proj,
        y_coords_img=model.y_coords_img,
        x_coords_img=model.x_coords_img,
        angles=model.angles)

    test_res = trainer.test(model, test_dl)[0]
    out_res = {
        "Mean PSNR": test_res["Mean PSNR"].item(),
        "SEM PSNR": test_res["SEM PSNR"].item()
    }
    with open('best_ckpt_results.json', 'w') as f:
        json.dump(out_res, f)


if __name__ == "__main__":
    main()
