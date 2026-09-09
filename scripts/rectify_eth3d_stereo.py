#!/usr/bin/env python3
"""Rectify ETH3D RGB stereo using camera 2/rgb2 as LEFT and camera 1/rgb as RIGHT.

ETH3D processed RGB images are already undistorted pinhole images, but rgb/rgb2
are not stereo rectified. extrinsics_1_2.txt maps camera-2 coordinates to
camera-1 coordinates. groundtruth.txt is camera-1 c2w; this script converts it
to the rectified camera-2/left viewpoint used by the stereo benchmark.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def K_from_file(path: Path) -> np.ndarray:
    fx, fy, cx, cy = np.loadtxt(path, dtype=np.float64).reshape(-1)
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)


def T_from_file(path: Path) -> np.ndarray:
    a = np.loadtxt(path, dtype=np.float64).reshape(3,4)
    T = np.eye(4); T[:3,:] = a
    return T


def q_to_R(qx,qy,qz,qw):
    q=np.array([qx,qy,qz,qw],dtype=np.float64); q/=np.linalg.norm(q)
    x,y,z,w=q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float64)


def R_to_q(R):
    # OpenCV gives Rodrigues but not quaternion; stable matrix->xyzw conversion.
    m00,m01,m02=R[0]; m10,m11,m12=R[1]; m20,m21,m22=R[2]; tr=m00+m11+m22
    if tr>0:
        s=np.sqrt(tr+1)*2; qw=.25*s; qx=(m21-m12)/s; qy=(m02-m20)/s; qz=(m10-m01)/s
    elif m00>m11 and m00>m22:
        s=np.sqrt(1+m00-m11-m22)*2; qw=(m21-m12)/s; qx=.25*s; qy=(m01+m10)/s; qz=(m02+m20)/s
    elif m11>m22:
        s=np.sqrt(1+m11-m00-m22)*2; qw=(m02-m20)/s; qx=(m01+m10)/s; qy=.25*s; qz=(m12+m21)/s
    else:
        s=np.sqrt(1+m22-m00-m11)*2; qw=(m10-m01)/s; qx=(m02+m20)/s; qy=(m12+m21)/s; qz=.25*s
    q=np.array([qx,qy,qz,qw]); q/=np.linalg.norm(q)
    if q[3]<0: q=-q
    return q


def pose_matrix(vals):
    T=np.eye(4); T[:3,:3]=q_to_R(*vals[4:8]); T[:3,3]=vals[1:4]
    return float(vals[0]),T


def pose_line(ts,T):
    q=R_to_q(T[:3,:3]); t=T[:3,3]
    return f"{ts:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"


def pad(img,w,h):
    ih,iw=img.shape[:2]
    if iw>w or ih>h: raise ValueError("image larger than common canvas")
    return cv2.copyMakeBorder(img,0,h-ih,0,w-iw,cv2.BORDER_CONSTANT,value=(0,0,0))


def valid_fraction(mx,my,w,h):
    return float(((mx>=0)&(mx<=w-1)&(my>=0)&(my<=h-1)).mean())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--alpha",type=float,default=0.0)
    ap.add_argument("--overwrite",action="store_true")
    args=ap.parse_args()
    src=Path(args.input).expanduser().resolve(); dst=Path(args.output).expanduser().resolve()
    if dst.exists() and args.overwrite: shutil.rmtree(dst)
    if dst.exists() and any(dst.iterdir()): raise RuntimeError(f"output exists and is non-empty: {dst}; use --overwrite")
    dst.mkdir(parents=True,exist_ok=True)

    left_dir,right_dir=src/"rgb2",src/"rgb"
    Kl,Kr=K_from_file(src/"calibration2.txt"),K_from_file(src/"calibration.txt")
    Trl=T_from_file(src/"extrinsics_1_2.txt") # camera2(left) -> camera1(right)
    R,T=Trl[:3,:3],Trl[:3,3]
    if not np.allclose(R.T@R,np.eye(3),atol=1e-5) or abs(np.linalg.det(R)-1)>1e-3:
        raise ValueError("invalid extrinsics_1_2 rotation")

    L={p.name:p for p in left_dir.glob("*.png")}; Rf={p.name:p for p in right_dir.glob("*.png")}
    names=list(set(L)&set(Rf))
    names.sort(key=lambda n: float(Path(n).stem))
    if not names: raise RuntimeError("no synchronized rgb2/rgb pairs")
    timestamps=[float(Path(n).stem) for n in names]
    if any(b<=a for a,b in zip(timestamps[:-1],timestamps[1:])): raise RuntimeError("non-increasing image timestamps")

    il=cv2.imread(str(L[names[0]]),cv2.IMREAD_COLOR); ir=cv2.imread(str(Rf[names[0]]),cv2.IMREAD_COLOR)
    if il is None or ir is None: raise RuntimeError("failed to read first pair")
    hl,wl=il.shape[:2]; hr,wr=ir.shape[:2]; w,h=max(wl,wr),max(hl,hr); size=(w,h)
    D=np.zeros(5,dtype=np.float64)
    R1,R2,P1,P2,Q,roi1,roi2=cv2.stereoRectify(Kl,D,Kr,D,size,R,T,flags=cv2.CALIB_ZERO_DISPARITY,alpha=args.alpha,newImageSize=size)
    K1,K2=P1[:,:3].copy(),P2[:,:3].copy()
    if not np.allclose(K1,K2,atol=1e-6,rtol=1e-7): raise RuntimeError("rectified K mismatch")
    baseline=float(np.linalg.norm(T)); signed=-float(P2[0,3])/float(P2[0,0])
    if signed<=0: raise RuntimeError(f"unexpected rectified baseline sign: {signed}; check LEFT=rgb2 RIGHT=rgb ordering")
    if abs(abs(signed)-baseline)>1e-4: raise RuntimeError("baseline mismatch after rectification")
    mx1,my1=cv2.initUndistortRectifyMap(Kl,D,R1,K1,size,cv2.CV_32FC1)
    mx2,my2=cv2.initUndistortRectifyMap(Kr,D,R2,K2,size,cv2.CV_32FC1)

    ol,orr=dst/"image_left",dst/"image_right"; ol.mkdir(); orr.mkdir()
    for i,n in enumerate(names):
        a=cv2.imread(str(L[n]),cv2.IMREAD_COLOR); b=cv2.imread(str(Rf[n]),cv2.IMREAD_COLOR)
        if a is None or b is None: raise RuntimeError(f"failed to read {n}")
        a=cv2.remap(pad(a,w,h),mx1,my1,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        b=cv2.remap(pad(b,w,h),mx2,my2,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        if not cv2.imwrite(str(ol/n),a) or not cv2.imwrite(str(orr/n),b): raise RuntimeError(f"failed to write {n}")
        if i%50==0 or i+1==len(names): print(f"[{i+1}/{len(names)}] {n}")
    (dst/"timestamps.txt").write_text("\n".join(Path(n).stem for n in names)+"\n")

    Trect_from_left=np.eye(4); Trect_from_left[:3,:3]=R1
    Tleft_from_rect=np.linalg.inv(Trect_from_left)
    gt=src/"groundtruth.txt"
    if gt.exists():
        with gt.open() as fi,(dst/"groundtruth_left.txt").open("w") as fo,(dst/"groundtruth_left_original_camera2.txt").open("w") as foo:
            for line in fi:
                line=line.strip()
                if not line or line.startswith("#"): continue
                vals=np.fromstring(line,sep=" ")
                if vals.size!=8: continue
                ts,Twr=pose_matrix(vals)
                Twl=Twr@Trl
                Twrect=Twl@Tleft_from_rect
                foo.write(pose_line(ts,Twl)+"\n"); fo.write(pose_line(ts,Twrect)+"\n")
        shutil.copy2(gt,dst/"groundtruth_right_original.txt")

    calib={
      "source":str(src),"camera_convention":{"left":"ETH3D camera 2 / rgb2","right":"ETH3D camera 1 / rgb","extrinsics_1_2":"T_right_original_from_left_original","groundtruth_left":"T_world_from_left_rectified"},
      "original_left_size":[wl,hl],"original_right_size":[wr,hr],"rectified_size":[w,h],
      "baseline_original_m":baseline,"baseline_rectified_m":abs(signed),"signed_baseline_rectified_m":signed,
      "K_left_original":Kl.tolist(),"K_right_original":Kr.tolist(),"T_right_original_from_left_original":Trl.tolist(),
      "R1":R1.tolist(),"R2":R2.tolist(),"P1":P1.tolist(),"P2":P2.tolist(),"Q":Q.tolist(),
      "K_rectified_left":K1.tolist(),"K_rectified_right":K2.tolist(),
      "T_rectified_left_from_original_left":Trect_from_left.tolist(),"T_original_left_from_rectified_left":Tleft_from_rect.tolist(),
      "roi_left":list(map(int,roi1)),"roi_right":list(map(int,roi2)),"alpha":float(args.alpha),
      "rectification_map_valid_fraction_left":valid_fraction(mx1,my1,w,h),"rectification_map_valid_fraction_right":valid_fraction(mx2,my2,w,h)}
    (dst/"calibration.json").write_text(json.dumps(calib,indent=2))
    print(f"Done: {dst}\nPairs={len(names)} size={w}x{h} baseline={abs(signed):.6f}m valid L/R={calib['rectification_map_valid_fraction_left']:.4f}/{calib['rectification_map_valid_fraction_right']:.4f}")


if __name__=="__main__": main()
