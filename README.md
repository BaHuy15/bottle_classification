- Move to yolov7_mask directory
    ```Shell
    cd ~/yolov7_mask
    ```

# Split and Convert data
- Format origin dataset: dataset only contains image files and their label files:
    - Image file: image_file_name.bmp
    - Label file: image_file_name.bmp.json
    </br>
    $\rightarrow$ Format in GLabbler
    
    - Exmaple about structure of the origin dataset:
    ```Shell
    data-name:
    |---image_name.bmp
    |---image_name.bmp.json
    ```
- Split dataset:
   ```Shell
   python convert_data/split_data.py \
   --data-path=path \
   --save-dir=path \
   --split-ratios=0.8 0.1 0.1 
   ```
   - <i>Where</i>:
        - <b>data-path</b>: path of your origin dataset
        - <b>save-dir</b>: path of new dataset which contains sub-datasets after origin dataset is split.
        - <b>split-ratios</b>: ratios of number of files split into <b>train, valid, test</b> sub-datasets respectively. And, total of split ratios is equal to 1.

   - After running CLI above, your origin dataset is split into <b>train, valid, test datasets</b> with <b>split_ratios</b> respectively.
   - Error images which have no label file moved to <b>error</b> folder.
   - Example about structure of the split dataset:
   ```Shell
   data-name
   |---train
   |---val
   |---test
   |---error
   ```
- Convert from GLabbeller format to COCO format:
    ```Shell
    python convert_data/convert_kla_format_multiclass.py \
    --data-path=path_of_dataset \
    --save-dir=path \
    --class-file=class_file_path
    ```
    - <i>Where</i>:
        - <b>data-path</b>: path of your dataset which has been just split.
        - <b>save-dir</b>: path of new dataset which contains converted data. And, it is organized for training your yolov7_mask model.
        - <b>class-file</b>: is a yaml file, contains all class names that you use.
            - Format for 'class_file.yaml':
            ```Shell
            names: [class_1, class_2, ..., class_n]
            ```
    - After running CLI above, your split dataset is organized and converted to COCO format to prepare for training. Beside, it also create a <b>data.yaml</b> which contains information about the new dataset and is used to refer argument --data for train, predict.
        - Example about structure of the new dataset:
        ```Shell
        data-name
        |---images
            |---train
            |---val
            |---test
        |---labels
            |---train
            |---val
            |---test
        |---data.yaml
        ```
        - Example about 'data.yaml':
        ```Shell
        train: save-dir/images/train
        val: save-dir/images/val
        test: save-dir/images/test

        nc: num_classes
        names: list_class_names
        ```

# Train CLI
```Shell
python3 seg/segment/train.py \
--weights=path_to_file_weights.pt \
--data=save-dir/data.yaml \
--hyp=seg/data/hyps/hyp.scratch-high.yaml \
--imgsz=image_size \
--epochs=num_epochs \
--batch-size=batch_size \
--device=device_id \
--project=root-path/project-name/train-seg
```

- Some augmentations that we can adjust to appropriate for training model:
```Shell
degrees: 0.0  # unit=deg, range=[0, 180] -> the image is rotated at random in range [-degrees, degrees]
translate: 0.1  # unit=fraction, range=[0, 1] -> the image is translated in range [-translate, translate]
scale: 0.9  # uint=gain, range=[0, 1] -> the image is scaled in range [-scale, scale]
shear: 0.0  # unit=deg, range=[0, 180] -> the image is sheared in range [-shear, shear]
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001 -> change 
flipud: 0.0  # unit=probability, range=[0, 1] -> image flip up-down
fliplr: 0.5  # unit=probability, range=[0, 1] -> image flip left-right
mosaic: 1.0  # unit=probability, range=[0, 1] -> image mosaic
mixup: 0.1  # unit=probability, range=[0, 1] -> image mixup
copy_paste: 0.1  # unit=probability, range=[0, 1] -> segment copy-paste
```
- Note:
    - mosaic: it resizes and concats 4 images into a large image, then crop the large image at random to create a new train image
    - mixup: mix 2 images with formula as follows: image = mixup * image1 + (1 - mixup) * image2
    - copy_paste: copy object from a image and past the object to another image
    - If mosaic is disabled (e.g mosaic=0), then mixup and copy_paste are also disabled



# Predict CLI:
```Shell
python3 seg/segment/predict.py \
--weights=path_to_file_weights.pt \
--source=path_to_image_or_image_dir \
--data=data_path.yaml \
--imgsz=image_size \
--conf-thres=confidence_threshold \
--iou-thres=iou_threshold \
--device=device_id \
--project=root-path/project-name/predict-seg
```

# Predict Under Kill and Over Kill
```Shell
python3 seg/segment/predict_UK_OK.py \
--weights=path_to_file_weights.pt \
--source=path_to_image_or_image_dir \
--source-label=path_to_label_or_label_dir \
--data=data_path.yaml \
--imgsz=image_size \
--conf-thres=confidence_threshold \
-iou-thresh=iou_threshold \
--device=device_id \
--project=root-path/project-name/predict-UK-OK-seg \
--UK-OK \
```

- Note:
    - If labels are bounding boxes, we add a argument <b>--gt-is-box</b> for this task
