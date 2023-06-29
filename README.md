- Move to yolov7_mask directory
    ```Shell
    cd ~/yolov7_mask
    ```

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