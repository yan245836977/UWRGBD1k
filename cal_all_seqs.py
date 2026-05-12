import os
import numpy as np
import shutil
from cal_iou_prec import *
import warnings
# 忽略所有DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
if __name__ == '__main__':
    trakcers = os.listdir("./pred_res/RGBX_SOT")

    wstr = "Methods, SR, PR, NPR\n"
    for tracker in trakcers:

        pd_rt = f"./pred_res/RGBX_SOT/{tracker}/"
        gt_rt = "I:/UW-RGBD1k/annotations/"

        seqs = [i.replace("\n","")+".txt" for i in open("I:/UW-RGBD1k/test_list.txt")]

        rgb_succ_score_all, rgb_prec_score_all, rgb_norm_prec_score_all = [], [], []
        for seq in seqs:
            pd_dir = pd_rt+seq
            gt_dir = gt_rt+seq


            try:
                try:
                    pd_bxs = np.loadtxt(pd_dir, delimiter="\t", dtype=np.int16)
                except:
                    pd_bxs = np.loadtxt(pd_dir, delimiter=" ", dtype=np.int16)
            except:
                pd_bxs = np.loadtxt(pd_dir, delimiter=",", dtype=np.int16)

            gt_bxs = np.loadtxt(gt_dir,delimiter="\t",dtype=np.int16)

            assert len(pd_bxs)==len(gt_bxs)
            protocol = 1
            # test RGB results
            rgb_succ_score, rgb_prec_score, rgb_norm_prec_score = calc_rgbps_seq_performace(pd_bxs,
                                                                                            gt_bxs,
                                                                                            protocol)

            rgb_succ_score_all.append(rgb_succ_score)
            rgb_prec_score_all.append(rgb_prec_score)
            rgb_norm_prec_score_all.append(rgb_norm_prec_score)

        rgb_succ_score = torch.tensor(rgb_succ_score_all).mean().tolist() * 100
        rgb_prec_score = torch.tensor(rgb_prec_score_all).mean().tolist() * 100
        rgb_norm_prec_score = torch.tensor(rgb_norm_prec_score_all).mean().tolist() * 100

        print(f'{tracker} | SR: {rgb_succ_score:.1f} | PR: {rgb_prec_score:.1f} | NPR: {rgb_norm_prec_score:.1f} | ')
        str = f"{tracker},{rgb_succ_score:.1f},{rgb_prec_score:.1f}, {rgb_norm_prec_score:.1f}\n"
        wstr+=str

    with open("mean_results/RGBX.csv","w") as f:
        f.write(wstr)