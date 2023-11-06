# Bottle classification using YOLOv5
  **(I'm currently updating this repo)**                                            
In this project, we applied yolov5 to classify 4 type of bottles:                               
- bottle_cap                           
- bottle_uncap                            
- fallen_bottle                      
- flip_bottle  

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Install Requirements](#Install-Requirements)
- [Data Format](#Data-Format)
- [Download Yolov7 Weights](#Download-Yolov7-Weights)
- [Evaluation](#Evaluation )
- [Training](#Training)
- [Result](#Result)
- [Citation](#Citation)
- [Acknowledgement](#Acknowledgements)

## Install Requirements
------------------------ --------------------
<details><summary> <b>Expand</b> </summary>                  
                                                                  
| Package | Version  |  
| :-- | :-: |
|absl-py|**1.4.0**|   
|albumentations |**1.3.0**|
|appdirs |  **1.4.4** |               
|attrs |  **22.2.0**  |
| backcall| **0.2.0**  |                 
| cachetools|  **5.3.0**  |               
| certifi| **2022.12.7**|                   
| chardet| **4.0.0** |
|charset-normalizer| **3.1.0** | 
|clearml| **1.10.3** |
| click| **8.1.3**|
| clip | **0.2.0**   |                                      
| cycler|**0.11.0** |                             
| decorator|**5.1.1**|            
| docker-pycreds|**0.4.0**  |                      
| fonttools| **4.38.0** |                              
| furl |**2.1.3**   |        
| gitdb | **4.0.10** |            
| GitPython |**3.1.31** |             
| google-auth| **2.17.2**|              
| google-auth-oauthlib| **0.4.6** |               
| grpcio | **1.53.0**  |               
| idna   | **3.4** |                                       
| imageio | **2.27.0** |                 
| imgaug | **0.4.0** |                         
| importlib-metadata| **6.3.0** |                      
| importlib-resources| **5.12.0** |                       
| ipython| **7.34.0**|          
| jedi | **0.18.2**  |                     
| joblib | **1.2.0**  |                
| jsonschema| **4.17.3**|                
| kiwisolver| **1.4.4** |                    
| Markdown | **3.4.3** |             
| MarkupSafe| **2.1.2** |                                     
| matplotlib | **3.5.3** |                   
| matplotlib-inline| **0.1.6** |                      
| networkx | **2.6.3** |                        
| numpy | **1.21.6**|                         
| nvidia-cublas-cu11 | **11.10.3.66**|                                   
| nvidia-cuda-nvrtc-cu11| **11.7.99** |                                                      
| nvidia-cuda-runtime-cu11| **11.7.99** |                                       
| nvidia-cudnn-cu11| **8.5.0.96** |                             
| oauthlib| **3.2.2** |                                            
| opencv-python |**4.7.0.72** |                            
| opencv-python-headless| **4.7.0.72** | 
| orderedmultidict| **1.0.1** |                     
| packaging | **23.0** |                    
| pandas |**1.3.5**|                  
| parso |**0.8.3**|                                        
| pathlib2 |**2.3.7.post1**|                        
| pathtools|**0.1.2** |     
| pexpect |**4.8.0**|                              
| pickleshare |**0.7.5**|           
| Pillow|**9.5.0**|                  
| pip|**23.0.1** |                
| pkgutil_resolve_name| **1.3.10**|                  
| prompt-toolkit|**3.0.38**|                                                     
| protobuf|**3.20.1**|                         
| psutil|**5.9.4**|                            
| ptyprocess|**0.7.0**|                                
| pyasn1|**0.4.8**|                 
| pyasn1-modules|**0.2.8** |           
| pycocotools|**2.0.6** |                
| Pygments|**2.14.0**|               
| PyJWT |**2.4.0**|               
| pyparsing|**3.0.9**|               
| pyrsistent|**0.19.3**|             
| python-dateutil|**2.8.2** |         
| python-dotenv|**0.21.1**|           
| pytz |**2023.3**|               
| PyWavelets|**1.3.0**|                              
| PyYAML|**6.0** |                
| qudida|**0.0.4**|            
| requests|**2.28.2**|              
| requests-oauthlib|**1.3.1**|           
| requests-toolbelt|**0.10.1**|          
| roboflow|**1.0.3**|       
| rsa|**4.9**|                       
| scikit-image |**0.19.3**|                                                      
| scikit-learn |**1.0.2**|                        
| scipy|**1.7.3**|                    
| seaborn|**0.12.2**|                
| sentry-sdk|**1.19.1**|                  
| setproctitle|**1.3.2**|                                                                          
| setuptools|**47.1.0**|                
| shapely|**2.0.1** |               
| six|**1.16.0**|                
| smmap|**5.0.0**|              
| tensorboard|**2.11.2**|             
| tensorboard-data-server|**0.6.1** |          
| tensorboard-plugin-wit|**1.8.1** |           
| thop |**0.1.1.post2209072238**|                                        
| threadpoolctl |**3.1.0**|                              
| tifffile |**2021.11.2** |                        
| torch  |**1.10.1+cu102** |                              
| torch-tb-profiler|**0.4.1** |                                                   
| torchaudio |**0.10.1+cu102** |                                           
| torchvision |**0.11.2+cu102**|                               
| tqdm   |**4.65.0** |                                 
| traitlets|**5.9.0**|                                       
| typing_extensions |**4.5.0**|                        
| ultralytics |**8.0.110**|                              
| urllib3 |**1.26.15** |                         
| wandb|**0.14.2** |                                                       
| wcwidth |**0.2.6**|                                      
| Werkzeug |**2.2.3** |                                        
| wget  |**3.2**  |                   
| wheel |**0.40.0**|                        
| zipp |**3.15.0**| 
   
</details> 

## Data format

<details><summary> <b>Expand</b> </summary> 

``` shell 
bottle_classification 
    |
    |____command                             
    |
    |_____data_bottle_detection                                                                            
    |        |
    |        |________test 
    |        |          |______.png #image file
    |        |          |_______.png.json #Json file                          
    |        |
    |        |________train
    |        |          |______.png #image file
    |        |          |_______.png.json #Json file                          
    |        |
    |        |_____train_data                       
    |        |       |_________images
    |        |       |                                 
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

## Training

Start training                   

``` Shell  
# Move to seg directory 
cd ~/bottle_classification/seg                     
python3 segment/train.py --weights=/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/yolov5s-seg.pt  --data=/home/tonyhuy/bottle_classification/seg/data/coco.yaml --hyp=/home/tonyhuy/bottle_classification/seg/data/hyps/hyp.scratch-med.yaml  --epochs=300  --device=5 
```

