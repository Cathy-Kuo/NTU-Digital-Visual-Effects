python3 code/hdr.py data/statue.txt out/ --npoints 50 --l 30 --alignment
python3 code/tonemap.py
python3 code/bilateralFilter.py --hdr_filename out/radiance_map.hdr \
    --compression_factor 0.1 \
    --ldr_scale_factor 60 \
    --sigma_s 1 \
    --sigma_r 1 \
    --kernel_size 5