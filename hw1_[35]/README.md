# Environment

- Python 3.9.7
- matplotlib 3.2.2
- Pillow 8.4.0
- numpy 1.21.1
- opencv-python 4.5.4.60


# Usage

1. `chmod +x code/*`
2. `./code/run.sh` (you can set parameters here)
3. `./code/tonemap.bat` (run in windows 7 and tm_*.exe should in your system PATH)


# Output

## run.sh
- `radiance_map.hdr`: radiance map
- `radiance_map.jpg`: radiance map visualization
- `g.jpg`, `g_red.jpg`, `g_green.jpg`, `g_blue.jpg`: response curves
- `ldr-Drago.jpg`, `ldr-Mantiuk.jpg`, `ldr-Reinhard.jpg`: tone map images of 3 algorithm from cv2
- `ldr-bilateral.jpg`: tone map image of Bilateral Filter

## tonemap.bat
- `radiance_map-xxx.jpg`: tone map images for different algorithm