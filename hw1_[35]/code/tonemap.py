import cv2

filename = 'out/radiance_map.hdr'
radiance_map = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)

# Tonemap using Drago's method to obtain 24-bit color image
print("Tonemaping using Drago's method ... ")
tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
ldrDrago = tonemapDrago.process(radiance_map)
ldrDrago = 3 * ldrDrago
cv2.imwrite("out/ldr-Drago.jpg", ldrDrago * 255)
print("saved ldr-Drago.jpg")

# Tonemap using Reinhard's method to obtain 24-bit color image
print("Tonemaping using Reinhard's method ... ")
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
ldrReinhard = tonemapReinhard.process(radiance_map)
cv2.imwrite("out/ldr-Reinhard.jpg", ldrReinhard * 255)
print("saved ldr-Reinhard.jpg")

# Tonemap using Mantiuk's method to obtain 24-bit color image
print("Tonemaping using Mantiuk's method ... ")
tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(radiance_map)
ldrMantiuk = 3 * ldrMantiuk
cv2.imwrite("out/ldr-Mantiuk.jpg", ldrMantiuk * 255)
print("saved ldr-Mantiuk.jpg")