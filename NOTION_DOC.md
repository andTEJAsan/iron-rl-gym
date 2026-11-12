# Physical RL Gym for Construction System Research / Design

## 1. Physical Description of the Task
On a real plot of land, we have three physical machines working together to clear earth:
1. **An Overhead Drone:** Hovers above the site to map the terrain, locate the dirt pile/trench, and track the position of the vehicles below.
2. **An Excavator (JCB Digger):** Positioned next to the dig zone. Its job is to lower its arm, scoop soil from the ground, swing its cab toward the waiting truck, and dump the soil into the truck bed without hitting the truck.
3. **A Haul Vehicle (Pickup / Dump Truck):** Parks in a loading zone next to the excavator, waits until its bed is loaded with several scoops of dirt, and then drives away to the drop-off zone.

To make this repeatable, we break the physical job into a continuous **15-second "Scoop-and-Dump" cycle**:
$$\text{Position Arm} \longrightarrow \text{Scoop Dirt} \longrightarrow \text{Swing to Truck Bed} \longrightarrow \text{Dump Load}$$
Once the scoop cycles fill the truck bed, the truck is triggered to drive off.

---

## 2. What We Want to Train
Out of the box, the excavator and truck are standard human-operated vehicles with no built-in intelligence. We want to train **two AI layers** to replace manual human operation:

1. **The Low-Level Excavator Driver (Primary Policy):**
   * A neural network that acts as the virtual operator inside the excavator cab.
   * **Inputs (What it sees):** Current angles of the excavator arm (`swing, boom, stick, bucket`), a camera/depth view of the dirt pile in front of the bucket, and the 3D location of the pickup truck bed.
   * **Outputs (What it controls):** The 4 continuous joystick commands (`[-1, +1]`) sent 20 times per second ($20\text{ Hz}$) to move the arm and bucket smoothly.
2. **The High-Level Site Coordinator (Multi-Agent Planner):**
   * Uses the drone’s overhead map to decide *where* the excavator should dig next and *where* the truck should park to minimize swing time and avoid collisions.

---

## 3. How the Data Will Be Collected
Because training an AI from scratch on a real 8-ton excavator by trial-and-error is unsafe and too slow, we collect data from **two sources**: **Real-World Human Demonstrations on the Land** and **Physics Simulation**.

### A. Real-World Data Collection (On the Land)
While a human operator (or teleoperator) drives the JCB and truck during our initial site runs, we record 50–100 scoop-and-dump cycles using lightweight sensors mounted on the machines:
* **Joystick Movements (Actions):** Sensors (or an interior cab camera) logging how the human moves the excavator levers at each millisecond.
* **Excavator Arm Pose (Body State):** Cheap strap-on IMU motion sensors on the boom, stick, bucket, and cab (plus visual AprilTag markers on the arm) to track exact joint angles.
* **Local Depth / Cab View:** A stereo depth camera mounted on the front of the JCB cab looking at the bucket and the soil being scooped.
* **Overhead Drone View (Global State):** A drone hovering overhead looking straight down, tracking visual markers on the excavator and pickup truck bed to log their exact relative positions and how much dirt has been moved.

### B. Data Processing Pipeline (Turning Logs into Training Episodes)
Our data pipeline takes those raw video and sensor recordings and converts them into clean, time-synchronized **15-second episodes** (sampled at $20\text{ Hz}$). For every step, the pipeline automatically scores the action (**Reward Labeling**) based on:
* How full the bucket got during the scoop (via camera segmentation of soil in the bucket).
* How smoothly the arm moved toward the truck bed.
* Whether the dirt successfully landed inside the pickup truck bed (via overhead depth change) without the bucket getting dangerously close to the truck cab.

---

## 4. What Will Be Trained & How It Will Be Trained (High-Level)
Once the data pipeline produces standardized `(Observation, Action, Reward)` episodes, we train the excavator brain in **three stages**:

1. **Stage 1 — Imitation Learning (Behavior Cloning from Real Land Data):**
   * We first train the neural network using supervised learning on the 50–100 recorded human scoops from the real land.
   * The network learns: *"When the camera and arm angles look like $X$, output the joystick command $Y$ that the human driver used."* This gives us a safe baseline policy that already knows the rough motion of scooping and swinging.
2. **Stage 2 — Large-Scale RL in GPU Simulation (Real-to-Sim):**
   * We import the drone’s 3D terrain scan of our land and a 3D model of the JCB + truck into a physics simulator (**NVIDIA Isaac Lab / MuJoCo / Custom Gym**).
   * Inside the simulator, we spawn virtual excavators in parallel and train the policy using **PPO (Proximal Policy Optimization)**. Here, the AI practices thousands of scoops across different soil stiffnesses and truck parking positions, getting rewarded for fast, full scoops and penalized for collisions or jerky movements.
3. **Stage 3 — Offline RL Fine-Tuning & Deployment (Sim-to-Real):**
   * Finally, we combine the simulator-trained policy with our growing dataset of real-world land recordings using **Offline RL (e.g., Advantage-Weighted Regression / IQL / SAC)**.
   * Every time the excavator runs on the real land, the new data is fed back through the pipeline to fine-tune the model on real soil resistance and hydraulic delays—steadily transitioning the site from human-assisted demos to full autonomy.

### Model Backbone Options (VLA vs. Hierarchical RL)
* **End-to-End VLA with Action Chunking (e.g., $\pi_0$ / OpenVLA fine-tuning):** Takes multi-view RGB (cab camera + overhead drone), excavator joint proprioception, and task prompts (e.g., *"scoop trench A and load truck"*), outputting $1\text{ s}$ chunks of continuous joystick actions.
* **Hierarchical VLM + PPO Controller:** A vision-language backbone running at low frequency ($2\text{–}5\text{ Hz}$) extracts 3D dig/dump waypoints from camera feeds, while a lightweight MLP policy trained via RL runs at high frequency ($20\text{–}50\text{ Hz}$) to handle non-linear soil resistance and smooth hydraulic control.
