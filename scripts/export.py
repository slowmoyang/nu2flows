from pathlib import Path

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

import logging

import h5py
import hydra
import torch as T
from omegaconf import DictConfig

from mltools.mltools.hydra_utils import reload_original_config
from mltools.mltools.torch_utils import to_np
from src.models.nuflows import NuFlows


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="export.yaml")
def main(cfg: DictConfig) -> None:
    log.info("Loading run information")
    orig_cfg = reload_original_config(ckpt_flag="*best*")

    log.info("Loading best checkpoint")
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model: NuFlows = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location=device, weights_only=False)
    model.samples_per_event = cfg.samples_per_event
    log.info(f'{cfg.samples_per_event=}')
    log.info(f'{model.samples_per_event=}')

    # Switch settings for the export
    orig_cfg.datamodule.loader_conf.batch_size = cfg.batch_size

    # Load the test datasets and run the predictions
    datamodule = hydra.utils.instantiate(orig_cfg.datamodule)

    log.info("Instantiating the trainer")
    orig_cfg.trainer["enable_progress_bar"] = True
    orig_cfg.trainer["logger"] = False  # Prevents a lightning_logs folder
    trainer = hydra.utils.instantiate(orig_cfg.trainer)

    log.info("Running the prediction loop")
    outputs = trainer.predict(model=model, datamodule=datamodule)

    log.info("Combining predictions across dataset")
    keys = list(outputs[0].keys())
    comb_dict = {}
    for k in keys:
        outs = [o[k] for o in outputs]
        if outs[0].ndim == 1:
            outs = [o.unsqueeze(1) for o in outs]
        comb_dict[k] = T.vstack(outs)
        log.debug(f'{k}: {comb_dict[k].shape}')

    log.info("Saving Outputs")
    output_dir_path = Path("outputs")
    output_dir_path.mkdir(exist_ok=True, parents=True)
    output_file_path = output_dir_path / f"test-{model.samples_per_event}.h5"
    log.info(f"Saving to {output_file_path}")
    with h5py.File(output_file_path, "w") as file:
        for key, v in comb_dict.items():
            file.create_dataset(key, data=to_np(v))


if __name__ == "__main__":
    main()
