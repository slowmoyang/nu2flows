import logging
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
from lightning import LightningDataModule
from numpy.lib.recfunctions import structured_to_unstructured as stu
from torch.utils.data import DataLoader, Dataset

from mltools.mltools.torch_utils import train_valid_split
from src.datamodules.physics import change_from_ptetaphiE

log = logging.getLogger(__name__)


class H5Dataset(Dataset):
    """Dataset class for neutrino regression stored as H5 files."""

    def __init__(
        self,
        *,
        file_list: list,
        data_dir: str,
        n_per_file: int | None = None,
        table_name: str = "delphes",
        met_kins: str | list = "px,py",
        lep_kins: str | list = "px,py,pz,log_energy",
        jet_kins: str | list = "px,py,pz,log_energy",
        nu_kins: str | list = "px,py,pz",
    ) -> None:
        """Parameters
        ----------
        file_list:
            List of datasets to load
        data_dir:
            The location of the datafiles
        n_per_file:
            Maximum number of events to load from each file
        table_name:
            The name of the table in the hdf5 file
        met_kins:
            The vars to use for the lepton kinematics
        lep_kins:
            The vars to use for the lepton kinematics
        jet_kins:
            The vars to use for the jet kinematics
        nu_kins:
            The vars to use for the neutrino (target) kinematics
        """
        super().__init__()

        # Save attributes
        self.met_kins = met_kins.split(",") if isinstance(met_kins, str) else met_kins
        self.lep_kins = lep_kins.split(",") if isinstance(lep_kins, str) else lep_kins
        self.jet_kins = jet_kins.split(",") if isinstance(jet_kins, str) else jet_kins
        self.nu_kins = nu_kins.split(",") if isinstance(nu_kins, str) else nu_kins

        # Get the list of files to use for the dataset
        self.file_list = []
        for f in file_list:
            file = Path(data_dir, f)
            if not file.exists():
                raise ValueError(f"Can't find requested file: {file}")
            self.file_list.append(file)

        # All data loaded from the files will be stored in these tensors
        npf = n_per_file
        log.info(f"loading {npf} events from each files...")
        self.misc = []
        self.met = []
        self.lep = []
        self.jet = []
        self.nu = []
        for file in self.file_list:
            log.info(file.name)
            with h5py.File(file, "r") as f:
                table = f[table_name]
                self.misc.append(
                    np.vstack([table["njets"][:npf], table["nbjets"][:npf]]).T.astype("float")
                )
                self.misc_vars = ["njets", "nbjets"]
                self.met.append(stu(table["MET"][:npf]))
                self.lep.append(stu(table["leptons"][:npf]))
                self.jet.append(stu(table["jets"][:npf]))
                self.nu.append(stu(table["neutrinos"][:npf]))
                self.met_vars = table["MET"].dtype.names
                self.lep_vars = table["leptons"].dtype.names
                self.jet_vars = table["jets"].dtype.names
                self.nu_vars = table["neutrinos"].dtype.names
        self.misc = np.vstack(self.misc).astype(np.float32)
        self.met = np.vstack(self.met).astype(np.float32)
        self.lep = np.vstack(self.lep).astype(np.float32)
        self.jet = np.vstack(self.jet).astype(np.float32)
        self.nu = np.vstack(self.nu).astype(np.float32)
        log.info(f"{len(self.met)} events loaded")

        # Jets need to be padded so create a mask
        self.jet_mask = ~np.all(self.jet == 0, axis=-1)

        # Neutrinos are always ordered particle -> antiparticle, so drop pdgid
        self.nu = self.nu[..., [1, 2, 3]]
        self.nu_vars = [self.nu_vars[i] for i in [1, 2, 3]]

        # Ensure that the lepton array is particle, anti (just like neutrino)
        order = np.argsort(self.lep[..., -2])  # orders by charge
        order = np.expand_dims(order, -1)
        self.lep = np.take_along_axis(self.lep, order, axis=1)

        # convert to specified coordinates
        log.info("converting data to specified coordinates...")
        self.met, self.met_vars = change_from_ptetaphiE(self.met, self.met_vars, self.met_kins)
        self.lep, self.lep_vars = change_from_ptetaphiE(self.lep, self.lep_vars, self.lep_kins)
        self.jet, self.jet_vars = change_from_ptetaphiE(self.jet, self.jet_vars, self.jet_kins)
        self.nu, self.nu_vars = change_from_ptetaphiE(self.nu, self.nu_vars, self.nu_kins, n_dim=3)

    def __len__(self) -> int:
        return len(self.met)

    def __getitem__(self, idx: int) -> list:
        """Return dictionaries for the inputs and the targets."""
        inputs = {
            "misc": self.misc[idx],
            "met": self.met[idx],
            "leptons": self.lep[idx],
            "jets": (self.jet[idx], self.jet_mask[idx]),
        }
        targets = {"neutrino": self.nu[idx][0], "antineutrino": self.nu[idx][1]}
        return inputs, targets

    def get_input_dims(self) -> tuple:
        """Return the typical dimensions of a data sample."""
        return {
            k: v[0].shape[-1] if isinstance(v, tuple) else v.shape[-1]
            for k, v in self[0][0].items()
        }

    def get_target_dims(self) -> tuple:
        """Return the typical dimensions of a data sample."""
        return {
            k: v[0].shape[-1] if isinstance(v, tuple) else v.shape[-1]
            for k, v in self[0][1].items()
        }


class H5DataModule(LightningDataModule):
    def __init__(
        self,
        *,
        train_conf: Mapping,
        valid_conf: Mapping,
        test_conf: Mapping,
        loader_conf: Mapping,
    ) -> None:
        """The datamodule for providing dilepton information.

        Parameters
        ----------
        train_conf:
            Config for the training dataset class.
        test_conf:
            Config for the testing dataset class.
        loader_conf:
            Config for the pytorch dataloader.
        val_frac:
            Fraction of dataset held out for validaiton. Defaults to 0.1.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Initialise a mini test set for now to infer the dimensions of the samples
        mini_conf = deepcopy(test_conf)
        mini_conf["n_per_file"] = 2
        self.miniset = H5Dataset(**mini_conf)

    def setup(self, stage: str) -> None:
        if stage in {"fit", "validate"}:
            self.train_set = H5Dataset(**self.hparams.train_conf)
            self.valid_set = H5Dataset(**self.hparams.valid_conf)
            self.n_train_samples = len(self.train_set)
            self.n_valid_samples = len(self.valid_set)

        if stage in {"test", "predict"}:
            self.test_set = H5Dataset(**self.hparams.test_conf)
            self.n_test_samples = len(self.test_set)

    def input_dimensions(self) -> tuple:
        """Return the typical dimensions of a input sample."""
        return self.miniset.get_input_dims()

    def target_dimensions(self) -> tuple:
        """Return the typical dimensions of the target sample."""
        return self.miniset.get_target_dims()

    @property
    def model_kwargs(self):
        return dict(
            input_dimensions=self.input_dimensions(),
            target_dimensions=self.target_dimensions(),
        )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_conf, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, **self.hparams.loader_conf, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        self.hparams.loader_conf["drop_last"] = False
        return DataLoader(self.test_set, **self.hparams.loader_conf, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
