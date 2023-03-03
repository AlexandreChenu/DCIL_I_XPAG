# DCIL_I_XPAG
Implementation of the first version of the DCIL algorithm ([paper](https://arxiv.org/abs/2204.07404)) based on jax-based XPAG library. 

# Install 

1. Clone DCIL repo,

```sh
git clone https://github.com/AlexandreChenu/DCIL_I_XPAG.git
```

2. Create virtual environment dcil_env from environment.ylm,

```sh
cd DCIL_XPAG
conda env create --name dcil_env --file environment.yml
```

3. Clone + install XPAG (+ Jax),

```sh
git clone https://github.com/perrin-isir/xpag.git
cd xpag
git checkout 9ef7dd74b74fc71cee83c6a476adfebe4b977814
pip install -e .
```
Check this Repo for more instructions.

4. Clone + install maze or humanoid environments 

```sh
git clone https://github.com/AlexandreChenu/gmaze_dcil.git
```
OR

```sh
git clone https://github.com/AlexandreChenu/gfetch_dcil.git
```

OR 

```sh
git clone https://github.com/AlexandreChenu/ghumanoid_dcil.git
```

and 

```sh
cd <env_directory>
pip install -e .
```-e .
```

### Dependencies for the Fetch environment (from First Return Then Explore)

To run DCIL in the fetch environment, please clone Go-Explore repo:

```sh
git clone https://github.com/uber-research/go-explore.git
```

move to the robustified directory: 

```sh
cd go-explore/robustified
```

From there, clone uber-research/atari-reset: 

```sh
git clone https://github.com/uber-research/atari-reset.git atari_reset
```

Finally, update the PYTHONPATH to include robustified: 

```sh
export PYTHONPATH=$PYTHONPATH:path_to_goexplore/go-explore/robustified
```

with path_to_goexplore, the absolute path to the go-explore repo. 

# Run Dubins Experiment

```sh
python test_DCIL_I_XPAG_dubins.py --demo_path ./demos/dubins_convert/1.demo --save_path </path/to/save/path>
```
# Run Fetch Experiment

```sh
python test_DCIL_I_XPAG_fetch.py --demo_path ./demos/fetch_convert/6.demo --save_path </path/to/save/path>
```

# Run Humanoid Experiment

```sh
python test_DCIL_I_XPAG_humanoid.py --demo_path ./demos/humanoid_convert/1.demo --save_path </path/to/save/path> --eps_state 0.5 --value_clipping 1
```


# Visual logs produced in /path/to/save/path

- trajs_it_- : training rollouts + skill-chaining evaluation
- value_skill_-_it_- : value for x-y position of skill starting state for different orientations 
- transitions_- : sampled training transitions + segment between true desired goal and relabelled desired goal

