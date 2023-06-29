# python3 segment/train.py \
# --weights=/home/charleschinh/TOMO/yolov7_mask/yolov7-seg.pt \
# --data=/home/charleschinh/TOMO/TrainData/coco.yaml \
# --hyp=/home/charleschinh/TOMO/yolov7_mask/seg/data/hyps/custom_hyps.yaml \
# --epochs=300 \
# --device=2 \
# --project=/home/charleschinh/TOMO/TrayBOM/train-seg

# python3 segment/train.py \
# --weights=/home/charleschinh/TOMO/yolov7_mask/yolov7-seg.pt \
# --data=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/data/coco.yaml \
# --hyp=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/data/hyps/hyp.scratch-med.yaml\
# --epochs=300 \
# --device=3 \
# --project=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask

python3 segment/train.py \
--weights=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/yolov5s-seg.pt \
--data=/home/tonyhuy/bottle_classification/seg/data/coco.yaml \
--hyp=/home/tonyhuy/bottle_classification/seg/data/hyps/hyp.scratch-med.yaml \
--epochs=300 \
--device=5 \