# Windows Setup Guide

This document describes how to configure the local development environment for the **Fraud Detection AI** project on Windows.

---

# Software Requirements

* Git for Windows
* Anaconda / Miniconda
* Python (managed through Conda)
* Java 17 (Eclipse Temurin)
* Visual Studio Code
* JupyterLab
* PySpark

---

# Create the Conda Environment

```bash
conda create -n fraud-ai python=3.13
conda activate fraud-ai
```

---

# Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

# Register the Jupyter Kernel

```bash
pip install ipykernel

python -m ipykernel install --user \
    --name fraud-ai \
    --display-name "Python (fraud-ai)"
```

Restart VS Code after registering the kernel.

---

# Java Installation

Install **Eclipse Temurin JDK 17**.

Example installation path:

```text
C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot
```

---

# Configure JAVA_HOME

Create or edit the system variable:

```text
JAVA_HOME
```

Value:

```text
C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot
```

**Important**

Do **not** add `\bin` to the value of `JAVA_HOME`.

Correct:

```text
JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot
```

Incorrect:

```text
JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot\bin
```

---

# Update PATH

Ensure that `PATH` contains:

```text
%JAVA_HOME%\bin
```

---

# Verify Java

```bash
java -version
```

Expected output:

```text
openjdk version "17.x.x"
```

---

# Verify Python

```bash
python -c "import sys; print(sys.executable)"
```

Expected output:

```text
C:\Users\<username>\anaconda3\envs\fraud-ai\python.exe
```

---

# Verify PySpark

```bash
python -c "import pyspark; print(pyspark.__version__)"
```

---

# Test Spark

Create a file called `test_spark.py`:

```python
from pyspark.sql import SparkSession

print("Step 1: Import successful")

spark = (
    SparkSession
        .builder
        .appName("Fraud Detection AI")
        .getOrCreate()
)

print("Step 2: Spark started successfully")

print(spark.version)

spark.stop()

print("Step 3: Spark stopped")
```

Run:

```bash
python test_spark.py
```

Expected output:

```text
Step 1: Import successful
Step 2: Spark started successfully
4.2.0
Step 3: Spark stopped
```

Some warnings such as `winutils.exe` or `NativeCodeLoader` may appear on Windows. They are expected and do not prevent local development.

---

# Troubleshooting

## Spark hangs when creating SparkSession

Symptom:

```python
SparkSession.builder.getOrCreate()
```

does not finish or displays:

```text
The system cannot find the path specified.
```

Possible causes:

* `JAVA_HOME` is incorrectly configured.
* Jupyter is using the wrong Python kernel.
* PySpark is installed outside the active Conda environment.

Checks:

```bash
echo %JAVA_HOME%

where python

python -c "import sys; print(sys.executable)"

python -c "import pyspark; print(pyspark.__file__)"
```

---

# Project Status

Local environment successfully configured for:

* Git
* Conda
* JupyterLab
* PySpark
* Java 17
* Spark 4.2.0
