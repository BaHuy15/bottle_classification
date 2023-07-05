# Run training
python3 segment/train.py \
--weights=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/yolov5s-seg.pt \
--data=/home/tonyhuy/bottle_classification/seg/data/coco.yaml \
--hyp=/home/tonyhuy/bottle_classification/seg/data/hyps/hyp.scratch-med.yaml \
--epochs=300 \
--device=5 \
