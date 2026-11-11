"""Isolated official Meta MHR->SMPL conversion worker for Phase-2 evaluation."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/MHR-main")
sys.path.insert(0, "/root/MHR-main/tools/mhr_smpl_conversion")
import smplx
from conversion import Conversion
from mhr.mhr import MHR


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--input",required=True); parser.add_argument("--output",required=True); parser.add_argument("--smpl-model",required=True); args=parser.parse_args()
    vertices=np.load(args.input,mmap_mode="r")
    if vertices.ndim!=3 or vertices.shape[1:]!=(18439,3) or not np.isfinite(vertices).all(): raise ValueError(f"invalid MHR vertices {vertices.shape}")
    converter=Conversion(mhr_model=MHR.from_files(lod=1,device=torch.device("cpu")),smpl_model=smplx.SMPL(model_path=args.smpl_model).to("cpu"),method="pytorch")
    result=converter.convert_mhr2smpl(mhr_vertices=np.ascontiguousarray(vertices,dtype=np.float32)*np.float32(100.0),single_identity=True,return_smpl_vertices=True,return_smpl_parameters=False,return_fitting_errors=True)
    smpl=np.asarray(result.result_vertices,dtype=np.float32); errors=np.asarray(result.result_errors,dtype=np.float32)
    if smpl.shape!=(len(vertices),6890,3) or not np.isfinite(smpl).all() or not np.isfinite(errors).all(): raise ValueError("invalid conversion output")
    np.savez(args.output,vertices=smpl,errors=errors)


if __name__=="__main__": main()
