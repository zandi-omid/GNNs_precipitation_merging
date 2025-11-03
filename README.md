# 🌧️ GNN-Based Precipitation Retrieval

This repository contains the training framework for a **Graph Neural Network (GNN)** model that spatially interpolate daily precipitation using static (topography) and dynamic (ERA5 + IMERG) features with daily and 0.1 degree spatila and temporal resolution.  
The graph structure is generated from DEM and GHCN gauge data, and available as a downloadable file.

---

## 📦 Data Access

The prebuilt graph containing all features and labels (`graph_with_features_labels.pkl`, ~460 MB uncompressed) can be downloaded here:

🔗 [Box Link – Download Graph](https://arizona.box.com/s/087ympthpmqpumrgr4jx8ups499jn7ty)

Place it under: data/graphs/
