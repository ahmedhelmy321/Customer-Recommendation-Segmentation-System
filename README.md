# Customer-Recommendation-Segmentation-System
# 🛒 Enterprise Customer Recommendation & Segmentation System

An end-to-end Big Data and AI framework designed to process massive retail transaction datasets, segment customer behavior profiles, and generate personalized product recommendations. This project combines distributed computing via **PySpark (MLlib)** with Deep Learning via **TensorFlow (Autoencoders)**, all served through an interactive enterprise dashboard built with **Streamlit**.

---

## ✨ Features

- **Big Data Scalability**: Built using **PySpark** to seamlessly handle large-scale data ingestion, cleaning, and preprocessing (over 541K raw transactions processed into clean profiles).
- **Advanced Customer Segmentation**: Dual-clustering architecture utilizing **KMeans** for hard behavioral segments (Active/Inactive) and Gaussian Mixture Models (**GMM**) for soft, probabilistic customer loyalty membership tracking.
- **Deep Learning Collaborative Filtering**: A custom **TensorFlow Autoencoder** (256 → 128 → 64 → 128 → 256) designed to combat the 98.3% sparsity of the Customer-Product interaction matrix, producing top-5 novel recommendations.
- **Dynamic Marketing Automation**: Automated customized outreach copy generation and personalized discount distribution based on behavioral clustering thresholds.
- **Production-Ready Pipeline**: Complete lifecycle deployment containing multi-page analytical dashboard routes, custom shell scripts (`setup.bat`, `run.bat`), and temporary space diagnostics (`cleanup.bat`).

---

## 📂 Repository Structure

```bash
├── assets/
│   └── style.css                     # Custom UI styling configurations
├── utils/
│   └── data_loader.py                # Efficient PySpark session and artifact loaders
├── notebooks/
│   └── Recommendation_System_Pipeline.ipynb  # Core training, modeling, and Spark evaluation
├── app.py                            # Main multi-page Streamlit dashboard interface
├── setup.bat                         # Automated virtual environment and dependency setup
├── run.bat                           # Streamlit server runtime script with Spark configurations
├── cleanup.bat                       # Automated script to wipe temporary .crc, Jupyter & OS caches
├── requirements.txt                  # Pinned distribution dependencies
├── uv.lock                           # Strict environment lockfile
└── README.md                         # Documentation framework
