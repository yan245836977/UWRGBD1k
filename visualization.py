import os
import numpy as np
import shutil
from cal_iou_prec import *
import warnings
import cv2
# 忽略所有DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
if __name__ == '__main__':
    colors = {
        # 红色系
        "red": (0, 0, 255),  # BGR: (0,0,255) | RGB: (255,0,0) | 十六进制: #FF0000
        "crimson": (60, 20, 220),  # 深红色 | BGR: (60,20,220) | RGB: (220,20,60) | 十六进制: #DC143C

        # 绿色系
        "green": (0, 255, 0),  # BGR: (0,255,0) | RGB: (0,255,0) | 十六进制: #00FF00
        "lime": (0, 255, 127),  # 酸橙绿 | BGR: (0,255,127) | RGB: (127,255,0) | 十六进制: #7FFF00

        # 蓝色系
        "blue": (255, 0, 0),  # BGR: (255,0,0) | RGB: (0,0,255) | 十六进制: #0000FF
        "navy": (128, 0, 0),  # 藏青色 | BGR: (128,0,0) | RGB: (0,0,128) | 十六进制: #000080

        # 黄色系
        "yellow": (0, 255, 255),  # BGR: (0,255,255) | RGB: (255,255,0) | 十六进制: #FFFF00

        # 紫色系
        "purple": (128, 0, 128),  # BGR: (128,0,128) | RGB: (128,0,128) | 十六进制: #800080

        # 白色与黑色
        "white": (255, 255, 255),  # BGR: (255,255,255) | RGB: (255,255,255) | 十六进制: #FFFFFF
        "black": (0, 0, 0)  # BGR: (0,0,0) | RGB: (0,0,0) | 十六进制: #000000
    }

    color_space = {"uwrgbd1k_ep20":"yellow","CAFormer(train_all)":"purple","SeqTrack":"white","PUTrack":"blue","OKTrack":"green","UOSTrack":"lime"}
    line_space = {"uwrgbd1k_ep20":"-","CAFormer(train_all)":"-","SeqTrack":"--","PUTrack":"blue","OKTrack":"green","UOSTrack":"--"}

    rgb_trakcers = ["PUTrack","OKTrack","UOSTrack"]
    rgbx_trackers = ["uwrgbd1k_ep20","CAFormer(train_all)","SeqTrack"]

    seqs = [i.replace("\n","")+".txt" for i in open("I:/UW-RGBD1k/test_list.txt")]
    seqs_d = os.listdir("VisRes")
    for seq in seqs:
        seq = seq.replace(".txt","")
        if seq in seqs_d:
            continue
        img_dir = f"I:/UW-RGBD1k/sequences/{seq}/"
        imgs = [img_dir+i for i in os.listdir(img_dir)]
        gt_rt = f"I:/UW-RGBD1k/annotations/{seq}.txt"
        gt_bxs = np.loadtxt(gt_rt,delimiter="\t",dtype=np.int16)


        assert len(gt_bxs)==len(imgs)

        wt_dir = f"VisRes/{seq}/"
        os.makedirs(wt_dir,exist_ok=True)

        for i in range(len(gt_bxs)):
            img_rt = imgs[i]
            name = img_rt.split("/")[-1]
            im_ori = cv2.imread(img_rt)
            gt = gt_bxs[i]

            im = cv2.rectangle(im_ori,gt,colors["red"],2)

            for tkr in rgb_trakcers:
                pd_dir = f"./pred_res/RGB_SOT/{tkr}/{seq}.txt"

                try:
                    try:
                        pd_bxs = np.loadtxt(pd_dir, delimiter="\t", dtype=np.int16)
                    except:
                        pd_bxs = np.loadtxt(pd_dir, delimiter=" ", dtype=np.int16)
                except:
                    pd_bxs = np.loadtxt(pd_dir, delimiter=",", dtype=np.int16)

                assert len(pd_bxs)==len(gt_bxs)

                pd = pd_bxs[i]
                clr = colors[color_space[tkr]]
                im = cv2.rectangle(im, pd, clr, 2)

            for tkr in rgbx_trackers:
                pd_dir = f"./pred_res/RGBX_SOT/{tkr}/{seq}.txt"

                try:
                    try:
                        pd_bxs = np.loadtxt(pd_dir, delimiter="\t", dtype=np.int16)
                    except:
                        pd_bxs = np.loadtxt(pd_dir, delimiter=" ", dtype=np.int16)
                except:
                    pd_bxs = np.loadtxt(pd_dir, delimiter=",", dtype=np.int16)

                assert len(pd_bxs) == len(gt_bxs)

                pd = pd_bxs[i]
                clr = colors[color_space[tkr]]
                im = cv2.rectangle(im, pd, clr, 2)

            cv2.imshow(f"Visualized {seq}", im)
            cv2.waitKey(1)

            wt_rt = wt_dir+name
            (h,w,c) = im.shape
            im = im[:,:w//2,:]
            cv2.imwrite(wt_rt,im)
        cv2.destroyAllWindows()
