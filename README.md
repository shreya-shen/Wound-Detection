# Wound Detection and Digital Band-aid Application

This program automatically detects wounds on arms in photographs and applies digital band-aids to cover them.

## Solution Overview

This project implements an automated wound detection and digital treatment system using computer vision techniques. The solution focuses on detecting red-colored wounds (cuts, scratches, bruises) on human skin and applying realistic digital band-aids to cover them.

**Problem-Solving Approach:**
The main challenge was distinguishing actual wounds from normal skin variations and other red objects. After testing multiple color detection methods (HSV, BGR, RGB ratios), the LAB color space proved most effective. The 'a' channel in LAB specifically represents the red-green color axis, making it ideal for detecting red wounds while filtering out pinkish skin tones.

**Key Techniques Used:**
- **LAB Color Space Analysis**: Used the 'a' channel with threshold > 160 to isolate red regions
- **Morphological Operations**: Applied opening and closing operations to clean noise and connect fragmented wound pixels
- **Contour Analysis**: Filtered detected regions by size (50-2000 pixels) and dimensions to eliminate false positives
- **Oriented Bounding Boxes**: Used cv2.minAreaRect() to determine wound orientation for proper band-aid alignment
- **Alpha Blending**: Implemented realistic band-aid application with transparency and edge feathering

**Libraries Used:**
- OpenCV: Core image processing and computer vision operations
- NumPy: Mathematical operations and array manipulation
- Matplotlib: Result visualization and user interface

**Assumptions:**
- Images contain human arms with visible skin
- Wounds appear as distinctly red regions compared to surrounding skin
- Images are well-lit with sufficient contrast between wounds and skin
- Wounds are between 50-2000 pixels in area (roughly 0.5-20 square centimeters)

## Features

- **Wound Detection**: Uses LAB color space segmentation to detect red wounds accurately
- **Smart Band-aid Placement**: Automatically scales, orients, and positions band-aids to naturally cover detected wounds
- **Natural Blending**: Uses alpha blending for realistic band-aid application
- **Visual Comparison**: Shows before and after images side by side

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the program:
```bash
python arm.py
```

2. Enter the path to your image when prompted
3. The program will process the image and display results
4. Choose whether to save the processed image

## How It Works

1. **Color Detection**: Uses LAB color space to detect red regions (wounds) with high precision
2. **Contour Analysis**: Filters detected areas by size and shape to identify actual wounds
3. **Band-aid Generation**: Creates realistic band-aid templates with proper coloring and texture
4. **Smart Placement**: Calculates optimal band-aid size and orientation using wound geometry
5. **Alpha Blending**: Blends the band-aid with the original image for realistic results

## Technical Details

- Uses LAB color space 'a' channel for red detection (threshold > 160)
- Applies morphological operations to clean up detection noise
- Filters contours by area (50-2000 pixels) and minimum dimensions (10x10)
- Creates adaptive band-aids that scale and rotate based on wound characteristics
- Uses oriented bounding boxes for precise wound orientation detection
- Implements alpha blending with rounded corners for natural appearance

## Limitations

- Works best with clear, well-lit images showing human arms
- Designed specifically for red-colored wounds (cuts, scratches)
- May have difficulty with very dark or very light skin tones
- Requires wounds to have sufficient red color contrast with surrounding skin
- Assumes wounds are roughly 0.5-20 square centimeters in size