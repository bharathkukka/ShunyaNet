# Different Ways I Trained My CNN Model

During this project, I realized that there isn’t just one “correct” way to train a machine learning model. Each method has its own advantages and limitations, and I ended up experimenting with multiple setups before finding what worked best for me.

## 1. Training with IDEs (VS Code, PyCharm)

I started by writing and training my CNN model directly inside IDEs like VS Code and PyCharm. These tools made it very easy to:
- Write and organize my code
- Debug errors step by step
- Quickly modify the model, hyperparameters, and data loading pipeline

However, when I actually started training the model, I noticed some practical issues:
- CNN training itself is compute-intensive and uses a lot of CPU/GPU and RAM.
- At the same time, the IDE was also consuming a significant amount of system memory and resources.

Because both the IDE and the training process were running together, my system became slow. The training took longer, the interface lagged, and sometimes the overall experience was frustrating. This made me realize that running heavy IDE applications during long training sessions is not the most efficient way to use my hardware resources.

From this, I learned an important lesson: **IDEs are great for development and debugging, but not always ideal for long, resource-heavy training runs.**

## 2. Training on Cloud with Google Colab

Next, I moved to online cloud environments like **Google Colab**. This was a big jump in terms of speed whenever I got access to a good **GPU** or **TPU**. Some clear advantages I saw:
- Training was much faster compared to my local CPU.
- I didn’t have to worry about my laptop overheating or slowing down.
- It was easy to install packages and set up the environment.

But after using Colab for a while, I also faced several issues:

1. **Dataset upload and loading:**  
   For good training speed, the dataset has to be uploaded to the Colab instance and usually unzipped into the Colab file system (e.g., `/content`). Uploading large datasets repeatedly is slow and annoying.

2. **Internet dependency:**  
   If the internet connection drops, the Colab session can disconnect. When that happens, the training process is stopped, and all progress from that run is lost unless I had manually saved the model weights.

3. **Unstable GPU/TPU sessions:**  
   GPU/TPU allocation is not always stable. Long training sessions (4–5+ hours) often disconnect due to:
   - Inactivity timeouts
   - Session limits
   - Backend machine reallocation

   When this happened, I had to restart the session, reload the dataset, and start training again from scratch (unless I had carefully saved checkpoints).

So while Colab is **very powerful for quick experiments and shorter runs**, it turned out to be unreliable for long, continuous training sessions for my CNN.

## 3. Training on My College Lab System (Local CPU + Terminal)

Finally, I switched to using my **college lab computer** for training:
- **CPU:** Intel i7 12th Gen
- **RAM:** 16 GB
- **Power:** UPS backup for power stability

Here, I changed my workflow slightly:
- I still used the IDE (like PyCharm) for **writing code, debugging, and fixing errors**.
- Once the script was working correctly, I **closed the IDE and other heavy applications**.
- Then I switched to **terminal-based training**:
  - Activated my local Python **virtual environment** (`.venv`).
  - Ran the training script from the system terminal (`main.py`).

Because the **dataset was stored locally** on the lab machine, there was:
- No need to upload or download data each time.
- No dependency on internet connectivity during training.

The **UPS power backup** also meant that even if there was a power cut, the system wouldn’t suddenly shut down, so long training runs were much safer.

This setup solved most of the problems I faced earlier:
- No random Colab disconnections.
- No extra RAM usage from heavy IDEs during training.
- Stable environment with consistent hardware.

In the end, I found that this approach—**developing and debugging in the IDE, then training from the terminal on a local machine with a proper setup**—was the most reliable and efficient way for me to train my CNN models for this project.

