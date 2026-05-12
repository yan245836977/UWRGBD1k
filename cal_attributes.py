import os
import numpy as np
import shutil
from cal_iou_prec import *
import warnings
import json

# 忽略所有DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == '__main__':
    trakcers = os.listdir("./pred_res/RGB_SOT")

    with open("attributes_count.json","r") as f:
        json_rec = json.load(f)
        att_names = list(json_rec.keys())

    with open("attributes.json","r") as f:
        attributes = json.load(f)

    sr_str = 'Methods, TO, OO, PO, HO, DS, SA, LD, DC, DD, DIS, SV, FM, DV, CM, MB, LI\n'
    pr_str = 'Methods, TO, OO, PO, HO, DS, SA, LD, DC, DD, DIS, SV, FM, DV, CM, MB, LI\n'
    npr_str = 'Methods, TO, OO, PO, HO, DS, SA, LD, DC, DD, DIS, SV, FM, DV, CM, MB, LI\n'

    for tracker in trakcers:
        tkr_sr = f'{tracker}, '
        tkr_pr = f'{tracker}, '
        tkr_npr = f'{tracker}, '

        pd_rt = f"./pred_res/RGB_SOT/{tracker}/"
        gt_rt = "I:/UW-RGBD1k/annotations/"

        seqs = [i.replace("\n","")+".txt" for i in open("I:/UW-RGBD1k/test_list.txt")]
        # OO_seqs = ""
        for att_n in att_names:

            rgb_succ_score_all, rgb_prec_score_all, rgb_norm_prec_score_all = [], [], []

            for seq in seqs:
                seq_n = seq.replace(".txt","")
                label = attributes[seq_n][att_n]
                if sum(label)==0:
                    continue

                label_idxs = [n for n, i in enumerate(label) if i==1]
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

                pd_bxs = pd_bxs[label_idxs]
                gt_bxs = gt_bxs[label_idxs]

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

            tkr_sr += f'{rgb_succ_score:.1f}, '
            tkr_pr += f'{rgb_prec_score:.1f}, '
            tkr_npr += f'{rgb_norm_prec_score:.1f}, '

        sr_str += f'{tkr_sr}\n'
        pr_str += f'{tkr_pr}\n'
        npr_str += f'{tkr_npr}\n'

        print(f"Calculating to {tracker}.")
    with open("attributes_results/RGB/attributes_SR.csv","w") as f:
        f.write(sr_str)

    with open("attributes_results/RGB/attributes_PR.csv","w") as f:
        f.write(pr_str)

    with open("attributes_results/RGB/attributes_NPR.csv","w") as f:
        f.write(npr_str)