# Training an RL agent on the snake game using proximal policy optimization

## Requirements
- Python 3.5
- [Tensorforce](https://github.com/reinforceio/tensorforce) (scripts without "ppo"; use latest version from Github)
- [OpenAI baselines](https://github.com/openai/baselines) (scripts with "ppo" in name)
- [OpenAI gym](https://github.com/openai/gym)
- pygame
- imageio (only to create gif movies)
- matplotlib (to display the learning curve)
- tensorflow

## Usage
Each script starting with "train" corresponds to a separate model. By default, each script loads pre-trained parameters and displays the agent in action:
```
python train_XXX.py
```
To retrain a model from scratch (this might overwrite the saved parameters in the corresponding subfolder), use:
```
python train_XXX.py --train
```
To load pre-trained weights and save the first 3 episodes as a GIF-movie, use
```
python train_XXX.py --savegif
```
For a detailed description and evaluation of each model see the corresponding [blog post](https://deeprljungle.wordpress.com/2018/03/28/tackling-the-hyperparameter-jungle-of-deep-reinforcement-learning/). 
To display the learning curve of model XXX use
```
python analyze_learning_curve.py -d XXX/
```
