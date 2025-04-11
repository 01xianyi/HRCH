# Dataset

This document provides instructions for setting up the data required for the HRCH project.

## Required Datasets

The HRCH project requires two main datasets:

1. **IAPR TC-12** - A benchmark collection of images with annotations in several languages
2. **MIRFlickr-25K** - A collection of 25,000 images from Flickr with tags

## Data Setup Instructions

### Directory Structure

All data files should be placed in the `./HRCH/data/` directory following this structure:

HRCH/
└── data/
    ├── final_data/
    │   └── (IAPR TC-12 content)
    ├── mirflickr/
         └── (MIRFlickr-25K content)

### IAPR TC-12 Dataset

1. Download the IAPR TC-12 dataset from: 
   - Link: https://pan.baidu.com/s/1LuNtnZRieuJuk7gF64htrg
   - Extraction code: 0411

2. Extract and place the content as follows:
   - Place the `final_data` folder in `./HRCH/data/`
   - Place the three additional files in `./HRCH/data/`

### MIRFlickr-25K Dataset

1. Download the MIRFlickr-25K dataset from:
   - Link: https://pan.baidu.com/s/12EF9fnYxaz_tyunvuUNAxQ
   - Extraction code: 0411

2. Extract and place the content as follows:
   - Place the `mirflickr` folder in `./HRCH/data/`
   - Place the three additional files in `./HRCH/data/`
