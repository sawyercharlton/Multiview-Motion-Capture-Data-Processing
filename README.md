# Multiview-Motion-Capture-Data-Processing


[//]: # (## Result Samples)

[//]: # (- Some current result samples can be found in `asset/`)

[//]: # ()
[//]: # (## Requirements)

[//]: # (See `requirements.txt`)

## Instructions
- Hyperparameters can be found in `src/configs/`

## Examples
 - Draw grid to get a reference for crop the video if needed
    ```ruby
    python draw_grid.py
    ```
 - Input the crop location and crop the video if needed
    ```ruby
    python crop_video.py
    ```
 - Calculate extrinsic parameters
    ```ruby
    python calc_extrinsic.py
    ```
 - Visualize stereo detected chessboard 
    ```ruby
    python vis_chessboard.py
    ```
## Reference
- https://github.com/MobileRoboticsSkoltech/RecSync-android

## Acknowledgements
