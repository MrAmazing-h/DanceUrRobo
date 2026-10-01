"""Reject incompatible model layouts before using the validated Vega profile."""

import json
from pathlib import Path


def validate_model(model):
    expected = json.loads((Path(__file__).parent / "data/model_contract.json").read_text())
    names = [model.joint(i).name for i in range(model.njnt)]
    if (
        model.nq != expected["nq"]
        or model.nv != expected["nv"]
        or names != expected["joint_names"]
        or model.jnt_qposadr.tolist() != expected["qpos_addresses"]
    ):
        raise ValueError(
            "Incompatible model layout. Use the released Vega-1P/F5D6 model; nq alone is insufficient."
        )
