# HalFSAM (MICCAI 2025, Oral)
SAM-based Haustral Fold Detection In Colonoscopy with Debris Suppression and Temporal Consistency

**C3VDv2 Dataset Link**: https://durrlab.github.io/C3VDv2/
## Updates
**2025-09-19:** Code and dataset are coming soon!
## Abstract
Haustral folds can serve as important landmarks to localize and navigate colonoscopes through the colon. Fold edges can be utilized for tracking in 3D reconstruction algorithms to generate colonoscopy coverage maps and ultimately reduce missed lesions. Current haustral fold detection models struggle with debris-filled colonoscopy videos and fail to maintain high temporal consistency due to their single-frame input. We introduce **HalF-SAM**, a **Ha**ustra**l** **F**old detection model utilizing the Segment Anything Model (**SAM**) image encoder, which suppresses edges from specular reflection and fecal debris. The SAM2-based memory module enhances temporal consistency, which is essential for tracking. Our experiments have shown significant improvements in haustral fold extraction accuracy and stability. We also release a training dataset with automatically annotated haustral fold edges in debris-filled high-fidelity colon phantom videos.
