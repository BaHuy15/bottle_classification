

# Data format

<details><summary> <b>Expand</b> </summary> 

``` shell 
bottle_classification 
    |
    |____command                             
    |
    |_____data_bottle_detection                                                                            
    |        |
    |        |________test                                       
    |        |
    |        |________train                              
    |        |
    |        |_____train_data                       
    |        |       |_________images                                         
    |        |       |_________labels                        
    |        |                                 
    |        |                       
    |        |                               
    |        |
    |        |______seg                      
    |        |        
    |        |                             
    |        |                        
    |        |        
    |        |                           
    |        |                                     
    |        |        
    |        |                             
    |        |
    |        |______predict_crop_data # images will be saved when run inference_blisters.py                                     
    |        |
    |        |______predict_test_data # images will be saved when run inference_blisters.py       
    |
    |______seg           
           |
           |            
           |          
           |
           |_____data 
           |        |        
           |        |_______hyps                          
           |        |         |_______coco.yaml                      
           |        |         |_______hyp.scratch-high.yaml                         
           |        |         |_______hyp.scratch-low.yaml                        
           |        |         |_______hyp.scratch-med.yaml 
           |        |                      
           |        |________scripts
           |        |________coco.yaml                  
           | 
           |_______models
           | 
           |_______runs 
           |               
           |_______segment
           |
           |_______utils

```                                             

</details> 

# Training

Start training                   

``` Shell  
# Move to seg directory 
cd ~/bottle_classification/seg                     
python3 segment/train.py --weights=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/yolov5s-seg.pt  --data=/home/tonyhuy/bottle_classification/seg/data/coco.yaml --hyp=/home/tonyhuy/bottle_classification/seg/data/hyps/hyp.scratch-med.yaml  --epochs=300  --device=5 
```

