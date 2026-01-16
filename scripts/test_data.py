import copy
import argparse
from pathlib import Path
from src.datamodules.dilepton import H5DataModule

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_file",
    type=Path,
)
args = parser.parse_args()

data_file = args.data_file
if not data_file.exists():
    raise FileNotFoundError(f"Data file {data_file} does not exist.")
if not data_file.is_file():
    raise ValueError(f"Data file {data_file} is not a file.")

data_dir = str(args.data_file.parent)



train_conf = dict(
    data_dir=data_dir,
    met_kins="px,py",
    lep_kins="px,py,pz,log_energy",
    jet_kins="px,py,pz,log_energy",
    nu_kins="px,py,pz",
    file_list=[
        data_file.name,
    ],
)

valid_conf = copy.deepcopy(train_conf)
test_conf = copy.deepcopy(train_conf)

datamodule = H5DataModule(
    train_conf=train_conf,
    valid_conf=valid_conf,
    test_conf=test_conf,
    loader_conf=dict(
        pin_memory=True,
        batch_size=64,
        num_workers=4,
    )
)

datamodule.setup(stage='fit')
data_loader = datamodule.train_dataloader()

for batch in data_loader:
    print(batch)
    break
