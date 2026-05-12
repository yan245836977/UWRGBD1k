# UWRGBD1k: a large-scale RGBD dataset of underwater object tracking

This work releases a new RGB and depth (RGBD) dataset of underwater objects named UWRGBD1k for implementing  underwater object tracking. The UWRGBD1k is a large-scale dataset that consists of 1228 video sequences and collects total about 640 k frames. Compared with current RGBD dataset photographed on land, UWRGBD1k mainly focus on marine creatures and includes more than 26 types of marine creatures. Welcome to use our data and cite our work!

## How to use UWRGBD1k
You can access our UW-RGBT1k via the [BaiduNet link]( https://pan.baidu.com/s/1R_9LVZV5B4z0idDm6Y0AjQ) (sjjb)

Dataset formal:
```python
UW-RGBD1k:
	-annotations # Bounding boxes are recorded in [x,y,w,h] for the object.
		-s0001.txt
		-s0002.txt
		-s0003.txt
		...
	-sequences # Image sequences are saved in individual frames (dim000000.jpg-dim***.jpg).
		-s0001
			-dim000000.jpg
			-dim000001.jpg
			-dim000002.jpg
			...
		-s0002
			-dim000000.jpg
			-dim000001.jpg
			-dim000002.jpg
			...
		-s0003
			-dim000000.jpg
			-dim000001.jpg
			-dim000002.jpg
			...
		...
	-visible_labels # Visible labels provide object visibilities in individual frames.
		-s0001.txt
		-s0002.txt
		-s0003.txt
		...
	-train_list.txt # Splited list of training sequences.
	-test_list.txt # Splited list of testimg sequences.
```
You can download this dataset and put it into your model.

## Evaluating tracking performances via our toolkits.
Our calculated toolkit can be found as:

```python
root:
	-attributes_results # Folder of attributed-evaluation results.
	-mean_results # Folder of overview-evaluation results.
	-pred_res # Folder of tracking results.
	-VisRes # Folder of visualized-evaluation results.
	-attributes.json 
	-attributes_count.json
	-cal_all_seqs.py
	-cal_attributes.py
	-cal_iou_prec.py
	-visualization.py
```

For the overview evaluation, you can run the cal_all_seqs.py:
```python
if __name__ == '__main__':
 	# You should put your tracking result (.txt) into a single folder and put this folder into the "./pred_res/RGBX_SOT"
    trakcers = os.listdir("./pred_res/RGBX_SOT")

    wstr = "Methods, SR, PR, NPR\n"
    for tracker in trakcers:

        pd_rt = f"./pred_res/RGBX_SOT/{tracker}/"
        gt_rt = "I:/UW-RGBD1k/annotations/" # Changing the direction with your saved place.
```

For the attributed evaluation, you can run the cal_attributes.py:
```python
if __name__ == '__main__':
 	# You should put your tracking result (.txt) into a single folder and put this folder into the "./pred_res/RGBX_SOT"
    trakcers = os.listdir("./pred_res/RGBX_SOT")

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

        pd_rt = f"./pred_res/RGBX_SOT/{tracker}/"
        gt_rt = "I:/UW-RGBD1k/annotations/" # Changing the direction with your saved place.
```


For the visualized evaluation, you can run the cal_attributes.py:
```python
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

    color_space = {"uwrgbd1k":"yellow","CAFormer(train_all)":"purple","SeqTrack":"white","PUTrack":"blue","OKTrack":"green","UOSTrack":"lime"}
    line_space = {"uwrgbd1k":"-","CAFormer(train_all)":"-","SeqTrack":"--","PUTrack":"blue","OKTrack":"green","UOSTrack":"--"}

    rgb_trakcers = ["PUTrack","OKTrack","UOSTrack"]
    rgbx_trackers = ["uwrgbd1k_ep20","CAFormer","SeqTrack"]

    seqs = [i.replace("\n","")+".txt" for i in open("I:/UW-RGBD1k/test_list.txt")]
    seqs_d = os.listdir("VisRes")
```
Citing format:

```python
@article{yan2026uwrgbd1k,
  title={UWRGBD1k: a large-scale RGBD dataset of underwater object tracking},
  author={Yan, Kaixiang and Qian, Wenhua and Bi, Cong and Liu, Peng},
  journal={Pattern Recognition},
  pages={113487},
  year={2026},
  publisher={Elsevier}
}
```
