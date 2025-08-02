# Multiview-Motion-Capture-Data-Processing
Process the output of [Synchronized-Video-Recorder-for-Android](https://github.com/yubohuangai/Synchronized-Video-Recorder-for-Android).

## Instructions
- Hyperparameters can be found in `config.yaml`

## Examples
 - Draw grid to get a reference for crop the video if needed
    ```ruby
    python draw_grid.py
    ```
 - Input the crop location and crop the video if needed
    ```ruby
    python crop_video.py
    ```
 - Extract frames and get matching information.
    ```ruby
    python match3.py
    ```
 - Visualize frame matching. 
    ```ruby
    python match_vis_3.py
    ```
 - Stitch the matched frames
    ```ruby
    python stitch3.py
    ```

## Acknowledgements
Yubo Huang