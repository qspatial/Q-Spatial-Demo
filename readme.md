# Quantum Lab Pre-Installation Guide

Quick setup guide for implementing QUBO, BQM, DQM, CQM, and Non-Linear (NL) Solver solutions on D-Wave quantum annealer using PyCharm IDE (recommended).

---

## 1. System Requirements


- **Python 3.9-3.11** (required) or later, you can get it on https://www.python.org/downloads/
- **PyCharm** (recommended) (Community or Professional Edition), you can get it on https://www.jetbrains.com/pycharm/download/
- pip Python's package manager

---

## 2. D-Wave Ocean SDK Installation

### 2.1 Installation

#### 2.1.1 Create PyCharm Project

(a) Open PyCharm with administrator privilages (run as administrator).

(b) Go to **File → New Project** 

(c) Set the project name to **"QuantumLab"** and the Python version, then click **Create** button.

#### 2.1.2 Open A Terminal inside PyCharm

(a) Navigate to **View > Tool Windows > Terminal**. This will open the Terminal tool window at the bottom of the PyCharm interface.

(b) Alternatively, **Press Alt + F12 (on Windows/Linux) or Option + F12 (on macOS)**. This will toggle the visibility of the Terminal tool window. 

#### 2.1.3 Install Packages

Run the following in your terminal:

```bash
# Install D-Wave Ocean SDK
pip install dwave-ocean-sdk

# Install additional packages
pip install numpy scipy matplotlib networkx
```

#### 2.1.3 Verify Installation
In PyCharm Terminal, run:
```bash
pip show dwave-ocean-sdk
```

Make sure the terminal shows dwave-ocean-sdk name and version along with other information. 

If a message such as **"WARNING: Package(s) not found: dwave-ocean-sdk"** appears, repeat the commands ensuring your PyCharm app is running with administrator privilages.

### 2.2 Configure PyCharm Python Interpreter

Verify interpreter settings:
1. **File → Settings** (Windows/Linux) or **PyCharm → Preferences** (macOS).
2. **Python → Interpreter**.
3. Verify virtual environment is on.
4. Check that `dwave-ocean-sdk` appears in package list.

---

## 3. D-Wave Leap Account Setup

### 3.1 Create Account

1. Visit [cloud.dwavesys.com/leap](https://cloud.dwavesys.com/leap/)
2. Create account using the email address we sent a D-Wave project invitation to.

### 3.2 Get API 

1. Log in to your D-Wave Leap account on [cloud.dwavesys.com/leap](https://cloud.dwavesys.com/leap/)
2. In your Leap Dashboard, navigate to **Project** dropdown list, select "UC Riverside Hackathon 2025"
3. Under **Solver API Token**, click **Copy** button to copy your API token.

### 3.3 Configure API


In PyCharm Terminal, run:
```bash
dwave config create
```

Enter when prompted:
- **API Endpoint:** `https://cloud.dwavesys.com/sapi/`
- **Authentication Token:** [paste your token]
- **Default Solver:** [press Enter to skip]


### 3.4 Test Connection to D-Wave Cloud Service
In PyCharm Terminal, run:
```bash
dwave ping
```

Run in PyCharm: **Right-click → Run 'test_connection'**

If the connection is successful, a message similar to the following will appear. If the connection failed, recreate the dwave config again.

```bash
Using endpoint: https://cloud.dwavesys.com/sapi/
Using region: na-west-1
Using solver: Advantage_system4.1
Working graph version: 01d07086e1
Submitted problem ID: 27466b42-6d7d-4534-925c-bff4b4216af9

Wall clock time:
 * Solver definition fetch: 18918.058 ms
 * Problem submit and results fetch: 2768.843 ms
 * Total: 21686.900 ms

QPU timing:
 * post_processing_overhead_time = 1.0 us
 * qpu_access_overhead_time = 6555.46 us
 * qpu_access_time = 15860.54 us
 * qpu_anneal_time_per_sample = 20.0 us
 * qpu_delay_time_per_sample = 20.58 us
 * qpu_programming_time = 15783.56 us
 * qpu_readout_time_per_sample = 36.4 us
 * qpu_sampling_time = 76.98 us
 * total_post_processing_time = 1.0 us
```

---

## 4. Writing and Running Code on D-Wave Leap

For each Python file we provided for this quantum lab exercise, repeat the following steps.

### 4.1 Create Python file

In PyCharm, right click on the project folder (we called it **"QuantumLab"**), navigate to **New > Python File**, set the file name, say **"OneScoopAtMost.py"**, and click **Enter**. This will create a Python file **OneScoopAtMost.py**.

### 4.2 Write or copy code in your file

Write or copy your code in the newly created file **OneScoopAtMost.py**. For this lab, copy the code we provided in the Python file and click **Save**.

### 4.2 Run your code

To run your code, **Right click** on the Python file **OneScoopAtMost.py** and click **Run OneScoopAtMost**. Your terminal will show the printed results.

---
