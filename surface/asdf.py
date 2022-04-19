import imageio
im = imageio.imread('./negative/00001.jpg')
imageio.imwrite('./astronaut-gray.jpg', im[:, :, 0])
