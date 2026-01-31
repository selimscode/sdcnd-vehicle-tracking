# Real-Time Vehicle Detection and Tracking

Computer vision pipeline for detecting and tracking vehicles in dashcam video using classical machine learning.

![Vehicle Detection Demo](media/project_video_out_gif.gif)

**[Watch full video on YouTube](https://youtu.be/ireg75njkQY)**

## Approach

This project uses a traditional ML pipeline (no deep learning) to detect vehicles:

1. **Feature Extraction** — HOG (Histogram of Oriented Gradients), color histograms, and spatial binning
2. **Classification** — Linear SVM trained on 17,000+ vehicle/non-vehicle images
3. **Sliding Window Search** — Multi-scale window search across the road region
4. **Temporal Filtering** — Heatmap accumulation across frames to reduce false positives

## Pipeline Visualization

| HOG Features | First Pass Detection | Heatmap | Final Result |
|:---:|:---:|:---:|:---:|
| ![HOG](media/output_12_1.png) | ![First Pass](media/output_44_2.png) | ![Heatmap](media/output_46_1.png) | ![Final](media/output_48_1.png) |

## Technical Details

### Feature Engineering
- **Color Space:** YCrCb (better vehicle/background separation than RGB)
- **HOG Parameters:** 9 orientations, 8 pixels/cell, 2 cells/block
- **Feature Vector:** ~8,000 dimensions combining HOG + spatial + color histogram

### False Positive Reduction
- Sliding window at multiple scales (1.0x, 1.5x, 2.0x)
- Heatmap thresholding across 5-frame buffer
- Minimum bounding box area filter (1500 px²)
- Weighted averaging favoring recent frames

### Performance
- **Training Accuracy:** ~99% on held-out test set
- **Processing:** ~2-3 FPS (not real-time, but suitable for offline analysis)

## Project Structure

```
├── vehicle_detection_pipeline.ipynb  # Full pipeline with visualizations
├── vehicle_detect.py                 # Vehicle detection module
├── lane_detect.py                    # Lane detection (bonus)
├── main.py                           # Video processing script
└── media/                            # Output videos and images
```

## Run It

```bash
# Install dependencies
pip install numpy opencv-python scikit-learn scikit-image matplotlib

# Run on video
python main.py --input project_video.mp4 --output output.mp4
```

## Future Improvements

- Replace SVM with CNN (YOLO/SSD) for real-time performance
- Add vehicle tracking with Kalman filter for smoother bounding boxes
- Extend to multi-class detection (cars, trucks, motorcycles, pedestrians)

## Context

Built as part of Udacity's Self-Driving Car Nanodegree. This project demonstrates classical computer vision techniques that were state-of-the-art before deep learning dominated object detection.

## Author

**Selim Cam** — [selimcam.dev](https://selimcam.dev) · [LinkedIn](https://linkedin.com/in/selimcam)
