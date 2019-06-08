# <b><u>Evaluating Self-Driving Perception Algorithms for Lane-Keeping in an Indoor Environment.</u></b>

## Preliminary Report Results:

[Visualisation of Comma's steering model I trained on their real data.](preliminary_results/comma-steering-prediction.gif)

[Testing Nvidia's steering model I trained on Udacity's simulator - Environment 1.](preliminary_results/udacity-steering-prediction-environment1.gif)

[Same model I trained on Udacity's simulator - Environment 2.](preliminary_results/udacity-steering-prediction-environment2.gif)

## Final Research Article: 

Applying modular and end-to-end techniques for lane-keeping to an indoor environment using an autonomous electric wheelchair. 

### Module Descriptions:
 - <i>bash_scripts:</i> contains scripts for communicating with the Raspberry Pis.
 - <i>data_capture_app:</i> this it the initial Android app used to collect steering angles. 
 - <i>docker_files:</i> contains Docker files for setting up a deep learning environment.
 - <i>final_results:</i> contains results of the evaluation process of the indoor environment and CARLA simulator.
 - <i>preliminary_results:</i> preliminary work completed as part of the Preliminary Research Article.
 - <i>src:</i> contains the Python code to collect data, train, test and evaluate lane-keeping algorithms.
 - <i>test_videos:</i> some test videos used in the process of refining the lane-keeping algorithms.

### Installation: 

 - Install the CARLA simulator from binary: 
    - https://github.com/carla-simulator/carla/blob/master/Docs/download.md
 - Install Python dependencies:

```bash
pip install -r requirements.txt
pip install git+https://github.com/carla-simulator/carla.git
```

### Usage:

Usage instructions can be found in the Python scripts.