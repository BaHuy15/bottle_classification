
#Bottle data
python3 seg/segment/predict.py 
--weights=/home/tonyhuy/bottle_classification/seg/runs/train-seg/exp7/weights/epoch_144_best.pt 
--source=/home/tonyhuy/bottle_classification/data_bottle_detection/home/pevis/TOMO_detection/data_bottle_detection/test  
--data=/home/tonyhuy/bottle_classification/seg/data/coco.yaml 
--imgsz=640  
--conf-thres=0.1  
--device=3

#Bottle data
# python3 seg/segment/predict.py 
# --weights=/home/tonyhuy/bottle_classification/seg/runs/train-seg/exp7/weights/epoch_144_best.pt 
# --source=/home/tonyhuy/bottle_classification/data_bottle_detection/Bottle_test 
# --data=/home/tonyhuy/bottle_classification/seg/data/coco.yaml 
# --imgsz=640  
# --conf-thres=0.1  
# --device=3

