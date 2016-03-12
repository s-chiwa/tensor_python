import cl_image
tnf = cl_image.TensorClassifyImage()
image = "test_data/cropped_panda.jpg"
#image = "test_data/img_37_3403_1.jpg"
print(tnf.run_inference_on_image(image))
