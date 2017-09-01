### How to run the files: ###

# 1.) Simply run the program and follow the on-screen prompts, entering the specified values to process an image
# 2.) If you would like to see the changes happening at each function, simply uncomment the 'cv2imshow' for that function.
# 3.) One window, 'Final result' , will show the final result and print the number of worms
# 4.) Whilst another window, 'Comparsion window' , will display the 'BBBC010_v1_foreground' ground_truth image
# 5.) The final window, 'Individual worms' , will allow the user to itterate through and view each individual worm
# 6.) To close all windows simply enter the 'x' key
# 7.) Once the windows have all been closed, help options will show in the shell
# 8.) Follow the on screen commands to call the help() functions to get extra information on each function
# 9.) Note- if running the straight from the python script, the help() commands may take a while to get through!

'''
    File name: IP summative.py
    Date created: 20/12/2016
    Date last modified: 19/01/2017
    Python Version: 2.7.10
    CVpython Version: 3.1.0
'''

__author__ = "pklm51"
__version__ = "1.0.1"
__maintainer__ = "pklm51"
__status__ = "Final draft"

import numpy as np
import cv2

    #####INITIALISATION#####

#reference images
img1_chan1 = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_B08_w1_CEBFFE77-AE3B-413C-BB2E-B0D1FC7A9F58.tif',-1)
img1_chan2 = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_B08_w2_E09DF96B-5EC9-4476-BFBB-9057C6D6C81C.tif',-1) 
 
img2_chan1 = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_D18_w1_EA00B7CA-24E0-46F2-AAA9-F0E900F17C68.tif',-1)
img2_chan2 = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_D18_w2_3EA4135F-9951-441F-BDF0-CE4C8C1305BC.tif',-1) 

img3_chan1 = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_D13_w1_3791F524-23BE-4B21-8F2A-9273C00FF13E.tif',-1)
img3_chan2 = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_D13_w2_ED5CEBD2-32D4-47FA-8A4F-F35880678E71.tif',-1) 


img1ExampleAnswer = cv2.imread('B08_binary.png',-1)                    
cv2.putText(img1ExampleAnswer,("Example ground_truth comparison"), (5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255,2)
img2ExampleAnswer = cv2.imread('D18_binary.png',-1)                     
cv2.putText(img2ExampleAnswer,("Example ground_truth comparison"), (5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255,2)
img3ExampleAnswer = cv2.imread('D13_binary.png',-1)                    
cv2.putText(img3ExampleAnswer,("Example ground_truth comparison"), (5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255,2)

    #####MANIPULATION#####

def manipulation1(img):         #NORMALIZE IMAGE
    cv2.normalize(img, img, 0, 2**16, cv2.NORM_MINMAX)                  #equilise between 0,2^16
    img = (img/256).astype(np.uint8)                                    #convert to 8-bit
    editedImage = img
    #cv2.imshow('normalize', editedImage);  
    return editedImage

def manipulation2chan1(img):    #BINARY- THRESHOLDING FOR CHANNEL 1                      
    ret2, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #applies the otsu threshold to the image
    img = cv2.erode(img,np.ones((3,3)),iterations = 1)                  #erodes the image for merging
    editedImage = img
    #cv2.imshow('binaryChan1', editedImage);
    return editedImage


def manipulation2chan2(img):    #BINARY- THRESHOLDING FOR CHANNEL 2                    
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,31,2)                                     #convert to binary
    img = (255-img)                                                     #invert image
    editedImage = img
    #cv2.imshow('binaryChan2', editedImage);
    return editedImage

def manipulation3(img):         #FIND CONTOURS AND REMOVE BORDER

    im2, contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    height,width = img.shape[:2]                                        #creates an array of contours
    blank = np.zeros((height,width),np.uint8)
    largestLength = 0
    CurrentLargest = None
    for i in contours:                                                  #for loop to find the largest worm
        length = cv2.arcLength(i,True)                                  #calculates the length of the current worm
        if length > largestLength:
            largestLength = length
            CurrentLargest = i
    cv2.drawContours(blank, [CurrentLargest], -1,255, 3)
    blank = cv2.dilate(blank,np.ones((35,35)),iterations = 1)           #colours in the largest worm(the border) 
    cv2.drawContours(img, [CurrentLargest], -1,0, 3)
    editedImgage = img - blank                                          #removes the border from the original image
    #cv2.imshow('remove border', editedImgage);
    return editedImgage


def manipulation4(img):         #REMOVE NOISE BY COLOURING BLACK
    img = cv2.medianBlur(img,3)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))    #morphs together neighbouring pixels
    im2, contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    height,width = img.shape[:2]
    blank = np.zeros((height,width),np.uint8)  
    for j in range (0,3):                                               #running 3 times ensures all noise is deleted
        im2, contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in contours:
            length = cv2.arcLength(i,True)
            if length < 100:                                            #any contour below the specified amount is assumed noise
                cv2.drawContours(img, [i], -1,0, 9)                     #colour the noise black
    editedImage = img
    #cv2.imshow('remove noise', editedImage);
    return editedImage

def manipulation5(img):         #COUNTS WORMS
    img[img<255]=0                                                      #ensures the image is truely binary
    im2, contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(img.shape,np.uint8)
    cv2.drawContours(blank, contours, -1,255, 3)  
    worms = 0
    for i in (contours):
        if cv2.arcLength(i,True) > 400:                                 #if the current wrom is over-lapping or a cluster
            worms = worms +2                                            #count both worms
        else:
            worms = worms +1                                            #accumulates worms
    print("Number of worms: " + str(worms))
    addText = img.copy()
    cv2.putText(addText,("Number of worms: " + str(worms)), (5,28), cv2.FONT_HERSHEY_SIMPLEX, 1, 255,3)
    cv2.imshow('final result', addText);                                #displays the final result   
    editedImage = img                                                   #without text
    return editedImage

def labelEachWorm(img):         #IDENTIFIES AND LABELS EACH WORM
    im2, contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(img.shape,np.uint8)
    cv2.drawContours(blank, contours, -1,255, -1)
    wormNumber = 0
    keepWindow = True                                                   #allows itteration through worms
    numberOfWorms = len(contours)
    while (keepWindow == True):
        currentWorm = cv2.minAreaRect(contours[wormNumber])             #finds height and width of each worm
        width, height = currentWorm[1][0],currentWorm[1][1]       
        blank = np.zeros(img.shape,np.uint8)                            #create a blank template to display the worm
        display = blank.copy()
        cv2.drawContours(blank, [contours[wormNumber]], -1,255, -1)     #choose the worm to display
        display = blank.copy()
        cv2.putText(display,("Worm number: " + str(wormNumber + 1) +  " (enter 'n' for next or 'p' for previous)"), (5,28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255,2)   
        if((width/height)> 6 or (height/width)> 6):
            cv2.putText(display,("Worm classification: dead"), (5,65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255,2)
        else:
            cv2.putText(display,("Worm classification: alive"), (5,65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255,2)
        cv2.imshow('Individual worms',display)
        key = cv2.waitKey(0)
        if (key == ord('x')):
            keepWindow = False
            cv2.destroyAllWindows()
        elif (key == ord('n')):                                         #displays next worm
            wormNumber +=1
            wormNumber %= numberOfWorms             
        elif (key == ord('p')):                                         #displays previous worm
            wormNumber -= 1
            wormNumber %= numberOfWorms
            

    #####FINALISATION#####

chooseTest = raw_input('Please enter which test image you would like to view (1,2 or 3): ') 

while (chooseTest != '1' and chooseTest != '2' and chooseTest != '3'):  #allows the user to decide which input image they wish to view
    chooseTest = raw_input('You have not entered a valid value. please enter 1, 2 or 3: ')

if chooseTest == '1':
    img1 = img1_chan1
    help1 = img1.copy()                                                 #saves a copy of the image at this current point for the help section
    img2 = img1_chan2
    print('you have decided to view image B08')
    cv2.imshow('Comparison window',img1ExampleAnswer)
    
if chooseTest == '2':
    img1 = img2_chan1
    help1 = img1.copy()                                                 #saves a copy of the image at this current point for the help section
    img2 = img2_chan2
    print('you have decided to view image D18')
    cv2.imshow('Comparison window',img2ExampleAnswer)
    
if chooseTest == '3':
    img1 = img3_chan1
    help1 = img1.copy()                                                 #saves a copy of the image at this current point for the help section
    img2 = img3_chan2
    print('you have decided to view image B13')
    cv2.imshow('Comparison window',img3ExampleAnswer)

if img1 is not None and img2 is not None:                               #ensures both channels have been input correctly

    FirstEditImg1 = manipulation1(img1)                                 #normalises the first channel
    help2 = FirstEditImg1.copy()                                        #saves a copy of the image at this current point for the help section
    FirstEditImg2 = manipulation1(img2)                                 #normalises the second channel
    help3 = FirstEditImg2.copy()                                        #saves a copy of the image at this current point for the help section

    SecondEditImg1 = manipulation2chan1(FirstEditImg1)                  #channel 1 and channel 2 get fed in to different function
    SecondEditImg2 = manipulation2chan2(FirstEditImg2)                  #the different functions help optimise the results for each channel

    MergedEdit = np.bitwise_or(SecondEditImg1,SecondEditImg2)           #merges both channels into one image
    help4 = MergedEdit.copy()                                           #saves a copy of the image at this current point for the help section
    
    ThirdEdit = manipulation3(MergedEdit)                               #removes the border from the image
    help5 = ThirdEdit.copy()                                            #saves a copy of the image at this current point for the help section
    FourthEdit = manipulation4(ThirdEdit)                               #removes noise and smooths worms
    help6 = FourthEdit.copy()                                           #saves a copy of the image at this current point for the help section
    FinalEdit = manipulation5(FourthEdit)                               #counts and prints the number of worms present
    help7 = FinalEdit.copy()                                            #saves a copy of the image at this current point for the help section

    print ("Close all windows, by typing 'x' on the window, to get extra information on each function")

    labelEachWorm(FinalEdit)                                            #create a window to display each different worm
     
else:
    print ("That image is not available")

print('[1] = Normalising function')
print('[2] = Thresholding on channel 1 with otsu')
print('[3] = Thresholding on channel 2 with thresholding')
print('[4] = Function for removing the border')
print('[5] = Function for removing noise')
print('[6] = Function for counting total worms')
print('[7] = Labelling and classification of worms function')


    ##### HELP #####

wantHelp = True                                                         #allows the user to view help for each different requested function
while (wantHelp == True):
    chooseHelp = raw_input('Would you like any more information/help about any particular function? (enter a value from above or "no" to escape): ')
    if chooseHelp == '1':
        help(manipulation1(help1))
    elif chooseHelp == '2':
        help(manipulation2chan1(help2))
    elif chooseHelp == '3':
        help(manipulation2chan2(help3))
    elif chooseHelp == '4':
        help(manipulation3(help4))
    elif chooseHelp == '5':
        help(manipulation4(help5))
    elif chooseHelp == '6':
        help(manipulation4(help6))
    elif chooseHelp == '7':
        help(labelEachWorm(help7))
    elif chooseHelp == 'no':
        wantHelp = False
    else:
        print('That is not a valid input.')                             #incorrect input handling

print('Thank-you for using the system.')

    

#some helpful functions:
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    #img = cv2.dilate(img,np.ones((1.5,1.5)),iterations = 1)
    #img = cv2.erode(img,np.ones((3,3)),iterations = 1)
    #maybe try cv2.floodFill(image, mask, seedPoint, newVal here
    #http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill
    
