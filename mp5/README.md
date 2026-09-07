# CS 444 / ECE 494: Deep Learning for Computer Vision, Fall 2025, Assignment 5

### Instructions

1. Assignment is due at **11:59:59 PM on Dec 4 2025**.

2. See [policies](https://saurabhg.web.illinois.edu/teaching/cs444/fa2025/policies.html) on [class website](https://saurabhg.web.illinois.edu/teaching/cs444/fa2025).

3. Submission instructions:
    1. On gradescope assignment called `MP5-code`, upload the following  files:
        - Your completed `vae.py`, `score.py`, `networks.py`, `diffusion_helpers.py` and `visual_anagrams.py` files.

       Please also note the following points:
        - Do not compress the files into `.zip` as this will not work.
        - Do not change the provided files names nor the names of the functions but rather change the code inside the provided functions and add new functions. Also, make sure that the inputs and outputs of the provided functions are not changed.
        - The autograder will give you feedback on how well your code did for the autograded parts.
    
    2. On gradescope assignment called `MP5-report`, upload the following
        -  Visualization + discussion of results from VAE and Score matching as mentioned in Q2.4
        -  Visualization + discussion  of results for different step sizes as mentioned in score matching Q2.5
        -  Visualizations for all parts in Q3
        -  Visual anagrams generated for the 2 prompts in Q4 along with one of your own choice of prompts



4. Lastly, be careful not to work of a public fork of this repo. Make a private clone to work on your assignment. You are responsible for preventing other students from copying your work.

    **⚠️ Disclaimer about equations**: 
    Web-based viewers (such as GitLab or GitHub web interfaces) may **silently drop** math symbols (like square roots) without any visual error. The equations may appear correct but be wrong. 

    To ensure you are implementing the correct formulas:
    - View the notebooks **locally** (VS Code, Jupyter Lab) or on the **Delta Cluster**.
    - If needed, cross-reference the equations with the **course slides**.


5. Change Log: 
    - Nov 15: Fixed the command to launch the jupyter notebook. Inserted semi-colon after setting the port number.
    - Nov 21: Clarified submission instructions for Q4 visual_anagrams, making the order of old man and campfire images consistent across instructions.
    - Nov 22: Added disclaimer on equation rendering by gitlab

### Suggested Development Workflow
This assignment contains four questions. Detailed instructions for each are provided in their respective notebooks.

1.  **Questions 1 & 2:** These questions do not require a powerful GPU and can be completed on a local machine.
2.  **Questions 3 & 4:** These questions involve loading a pre-trained text-conditioned diffusion model and require a GPU with approximately **7 GB of VRAM**. While we are not training a model, these questions involve generating images that benefits from GPU compute. You can make use of the following **compute options:** 
    1. **GPU-backed Jupyter Notebook on the Delta AI cluster (Recommended)** You can use a GPU-backed Jupyter Notebook on the Delta AI cluster. We have set up virtual environments with all the relevant dependencies for you to immediately get started on Delta AI. Virtual environment for Q1 and Q2 is at `/work/nvme/bfdu/cs444/fa25/mp5-q1-q2/venv/bin/activate` and virtual env for Q3 and Q4 is at `/work/nvme/bfdu/cs444/fa25/mp5-q3-q4/venv/bin/activate`. You can follow the following instructions to launch and use a jupyter notebook on Delta AI. Note that the job will terminate at the 1 hour mark. You can change this to a higher number if you like, **but make sure to save your work as you go, because you may not realize when the hour is over and your job is killed (and your notebook doesn't get saved)**.
        1. Get an interactive job:
            ```
            srun -A bfdu-dtai-gh --time=01:00:00 --nodes=1 --cpus-per-task=32 --partition=ghx4 --gpus=1 --mem=48g --pty /bin/bash
            ```
            If this command doesn't get you a job in a few minutes, try replacing `--partition=ghx4` by `--partition=ghx4-interactive`.
        2. Once the job launches, load the necessary modules and source the appropriate virtual env.
            ```
            module load python/3.10.14 cuda/12.4.0 gcc-native/12.3
            source /work/nvme/bfdu/cs444/fa25/mp5-q1-q2/venv/bin/activate
            ```
        3. Launch a Jupyter notebook. Replace TBD with a random 5 digit number less than 60000
            ```
            NB_PORT=TBD; jupyter notebook --ip='*' --port=$NB_PORT --port-retries=0 --no-browser \
              --NotebookApp.allow_origin='*' --NotebookApp.disable_check_xsrf=True \
              --NotebookApp.base_url=/node/$SLURM_JOB_NODELIST/$NB_PORT/ \
              --NotebookApp.custom_display_url=https://gh-ondemand.delta.ncsa.illinois.edu
            ```
        4. The above command will produce two URLs, one of the form `https://gh-ondemand...` and another of the form `https://127.0.0.1...`. Access your notebook through a browser using the `https://gh-ondemand...` URL. You will need to log in using your Delta AI credentials. Do not share this URL with anyone, they may be able to access your notebook!
        5. Make sure that you are using some way to secure access to your jupyter notebook. If you haven't configured jupyter any other way, it generates a token in the URL and the notebook can only be acessed through a URL that includes this token. You acn read up more about securing a jupyter notebook [here](https://jupyter-notebook.readthedocs.io/en/6.2.0/security.html), and make sure that your Jupyter Notebook is secured in some manner. It is also possible to disable all security features, but please do NOT do that.
        6. There is another way to use a jupyter notebook on Delta AI cluster, via port forwarding as explained in the [official documentation](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/software.html#how-to-run-jupyter-on-a-compute-node). You are welcome to also use that if you'd like. 
    2. **Jupyter Notebooks through the Delta AI Open OnDemand online interface** available via [this link](https://gh-ondemand.delta.ncsa.illinois.edu/). Similar to the previous option, it will run on Delta AI GPUs, but let's you get a notebook through a web interface, albiet with less control on what python it uses, and where the virtual environment is located, etc.
        1. Log in with your NCSA username and password (the same credentials used for SSH).
        2. After logging in, select the **Jupyter Lab** option. Configure your session by setting the `partition` to `ghx4` (or `ghx4-interactive`, if wait times are too long for `ghx4`) and selecting the appropriate path to your working directory. Make sure to only request a single GPU.
        3. You will need to set up your own development environment. We have provided `pip install` commands in the Notebooks.
        4. **Important:** Please note each interactive Jupyter session on DeltaAI is capped at **1 hour**. We recommend using them for debugging or final runs rather than for prolonged development. More details are available in the [DeltaAI documentation](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/ood/index.html). **Make sure to save your work as you go.**
    3. You can also use platforms like Google Colab, Kaggle, [Illinois Computes Notebooks](https://computes.illinois.edu/resources/icrn/) or even your personal machine (if you have a decent GPU that meets the spec). 
