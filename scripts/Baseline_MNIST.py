import argparse
import glob
import json
from os.path import exists

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from fit.datamodules.baselines import MNISTBaselineDataModule
from fit.modules.ConvBlockBaselineModule import ConvBlockBaselineModule


def main():
    seed_everything(28122020)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_config")

    args = parser.parse_args()

    with open(args.exp_config) as f:
        conf = json.load(f)

    dm = MNISTBaselineDataModule(root_dir=conf['root_dir'], batch_size=conf['batch_size'])
    dm.setup()

    model = ConvBlockBaselineModule(img_shape=dm.IMG_SHAPE,
                                    lr=conf['lr'], weight_decay=0.01,
                                    d_query=conf['d_query'])

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

    trainer.fit(model, datamodule=dm);

    model = ConvBlockBaselineModule.load_from_checkpoint('lightning_logs/version_0/checkpoints/best_val_loss_-last.ckpt')

    test_res = trainer.test(model, datamodule=dm)[0]
    out_res = {
        "Mean PSNR": test_res["Mean PSNR"].item(),
        "SEM PSNR": test_res["SEM PSNR"].item()
    }
    with open('last_ckpt_results.json', 'w') as f:
        json.dump(out_res, f)

    best_path = glob.glob('lightning_logs/version_0/checkpoints/best_val_loss_-epoch*')[0]
    model = ConvBlockBaselineModule.load_from_checkpoint(best_path)

    test_res = trainer.test(model, datamodule=dm)[0]
    out_res = {
        "Mean PSNR": test_res["Mean PSNR"].item(),
        "SEM PSNR": test_res["SEM PSNR"].item()
    }
    with open('best_ckpt_results.json', 'w') as f:
        json.dump(out_res, f)


if __name__ == "__main__":
    main()
