# python3 seg/segment/predict.py\
#     --weights="/home/bosco/Yolov7_TOMO/yolov7/seg/runs/train-seg/blister_val_702/weights/epoch_91_1.9.pt"\
#     --source="/home/jay2/Transfer/Peter/dataset/blister/test/test_multi_blister_images"\
#     --data="/home/jay2/Transfer/Peter/dataset/blister/test/test_blister_Finn_2022_11_02_coco/coco.yaml"\
#     --imgsz=640\
#     --conf-thres=0.35\
#     --device=3\

python3 seg/segment/predict.py\
--weights=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/runs/train-seg/exp7/weights/epoch_148_best.pt\
--source=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/Dataset/data_blister_10_11/test_blister_Finn_2022_11_02\
--data=/home/jay2/Transfer/Peter/dataset/blister/test/test_blister_Finn_2022_11_02_coco/coco.yaml\
--imgsz=640\
--conf-thres=0.35\
--device=3\