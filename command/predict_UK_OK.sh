# python3 yolov7_mask/seg/segment/predict_UK_OK.py \
# --weights=/home/charleschinh/TOMO/TrayBOM/train-seg/exp2/weights/epoch_223_best.pt \
# --source=/home/charleschinh/TOMO/TestData/images \
# --source-label=/home/charleschinh/TOMO/TestData/labels \
# --UK-OK \
# --data=/home/charleschinh/TOMO/TrainData/coco.yaml \
# --imgsz=640 \
# --conf-thres=0.1 \
# --device=2 \

# python3 seg/segment/predict_UK_OK.py \
# --weights=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/runs/train-seg/exp7/weights/epoch_148_best.pt \
# --source=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/Dataset/data_blister_10_11/train_data/images \
# --source-label=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/Dataset/data_blister_10_11/train_data/labels\
# --UK-OK \
# --data=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/data/coco.yaml \
# --imgsz=640 \
# --conf-thres=0.1 \
# --device=3 \
#yolov7_mask/

python3 seg/segment/predict_UK_OK.py \
--weights=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/runs/train-seg/exp7/weights/epoch_148_best.pt  \
--source=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/Dataset/data_blister_10_11/test_blister_Finn_2022_11_02 \
--UK-OK \
--data=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/data/coco.yaml\
--imgsz=640 \
--conf-thres=0.1 \
--device=3 \

python3 seg/segment/process_pred.py \
--weights=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/runs/train-seg/exp7/weights/epoch_148_best.pt  \
--source=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/Dataset/data_blister_10_11/test_blister_Finn_2022_11_02 \
--UK-OK \
--data=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/data/coco.yaml\
--imgsz=640 \
--conf-thres=0.1 \
--device=3 \
# python3 seg/segment/predict_UK_OK.py  --weights=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/runs/train-seg/exp7/weights/epoch_148_best.pt   --source=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/Dataset/data_blister_10_11/test_blister_Finn_2022_11_02  --UK-OK  --data=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/data/coco.yaml --imgsz=640  --conf-thres=0.1 \