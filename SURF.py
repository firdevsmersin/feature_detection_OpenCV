# # SURF (Speeded-Up Robust Features)


# ## Import resources and display image

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise
def affine_transformation(image):
    rows,cols,ch = image.shape
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst

def add_noise_to_image(image):
    noise_img = random_noise(image, mode='s&p',amount=0.3)
    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img
#classname_ =["butterfly","camera","crab","cup","dragonfly","flamingo","sunflower"]
classname_ = ["panda"]
filtername_ =["blur","rotate","flip","noise","affine"]
keypointname_ = ["detect","match","original"]
for classname in classname_:
    path="./%s/"%classname
    image_path = "./%s"%classname
    for filtername in filtername_:
        for keypointname in keypointname_:
            #keypointname = "original"
            file = open("%sSURF_%s_%s_%s.txt"%(path,classname,filtername,keypointname),"w+")
            for i in range(51):

                image1_='%s/image_000%d.jpg' % (image_path,1)
                if not i==0:
                    if i<10:
                        image1_='%s/image_000%d.jpg' % (image_path,i)
                    else:
                        image1_='%s/image_00%d.jpg' % (image_path,i)
                print(image1_+"\n")
                # Load the image
                image1 = cv2.imread(image1_)
                if filtername == "blur":
                     image2 = cv2.GaussianBlur(image1,(5,5),0)#add gaussian blur filter to image
                elif filtername == "rotate":
                     image2= cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
                elif filtername == "flip":
                     image2 = cv2.flip(image1, 1)#flip image horizontally
                elif filtername == "noise":
                    image2 = add_noise_to_image(image1)#add noise to image salt&pepper
                elif filtername == "affine":
                    image2 = affine_transformation(image1) # all parallel lines in the original image will still be parallel in the output image
                
                # Convert the training image to RGB
                training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

                # Convert the training image to gray scale
                training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)

                # Convert the test image to RGB
                test_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

                # Convert the training image to gray scale
                test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

                # ## Detect keypoints and Create Descriptor

                surf = cv2.xfeatures2d.SURF_create(800)

                train_keypoints, train_descriptor = surf.detectAndCompute(training_gray, None)
                test_keypoints, test_descriptor = surf.detectAndCompute(test_gray, None)

                keypoints_without_size = np.copy(training_image)
                keypoints_with_size = np.copy(training_image)

                file.write("\n")
                # Print the number of keypoints detected in the training image
                #print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))
                if keypointname =="original":
                    file.write(str(len(train_keypoints)))

                # ## Matching Keypoints

                # Create a Brute Force Matcher object.
                bf = cv2.BFMatcher()
                # Perform the matching between the SURF descriptors of the training image and the test image
                matches = bf.knnMatch(train_descriptor,test_descriptor,k=2)

                # Apply ratio test
                good = []
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        good.append([m])


                # Print the number of keypoints detected in the query image
                if keypointname == "detect":
                    print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))
                    file.write(str(len(test_keypoints)))
                elif keypointname == "match":
                    # Print total number of matching points between the training and query images
                    print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(good))
                    file.write(str(len(good)))
        