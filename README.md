# Sift based face recognition from ID cards

## Implementation

1. Read train and query images
2. Extract sift features
3. Calculate boundries using homograpy transform
4. find orintation of ID card
5. Rotate ID card and warp 
6. Run dlib face detector
7. Crop face regions
8. Extract sift features from cropped faces
9. Compare faces with Lowe's ratio  
## input image

![testcard](https://user-images.githubusercontent.com/47300390/151219450-d2562624-ec52-4666-8ea6-bf0d69e964c4.png)

## query images
![tc_ID](https://user-images.githubusercontent.com/47300390/151219528-15da8465-cec2-446a-825c-dd0eb8f10636.jpg)
![tc_ID_rot](https://user-images.githubusercontent.com/47300390/151219534-8376c282-2cc4-4679-8d13-a741e4ac61e9.jpg)

## rotated image
![rotated_img](https://user-images.githubusercontent.com/47300390/151220906-e0630a48-e986-4762-b961-150d40c82c3a.png)

## sift based recognition and warping
![warped_img](https://user-images.githubusercontent.com/47300390/151219708-c1a98897-867e-4a00-b16e-e816bdc1d28a.png)

## dlib face detection for both train and query image

![face_crop_target](https://user-images.githubusercontent.com/47300390/151219870-69f785af-0f12-47ab-89a2-a71136b4cf7d.png)

![cropped_fc](https://user-images.githubusercontent.com/47300390/151223859-fb9886f9-7011-4f8a-b5d5-82211b6681d7.png)

## matched points
![Figure_1](https://user-images.githubusercontent.com/47300390/151219748-92ca3625-4248-4edd-ac5f-6114c85a4523.png)

## matched ID cards
![matched_points_ID](https://user-images.githubusercontent.com/47300390/151219975-cb3dd920-1597-4246-a6d1-b485a33ebae0.png)



## Detection Results

1. Total good matches: 134
2. Rotation Angle: -43 degrees
3. Number of Faces: 1 (input image)
4. Number of Faces: 1 (output image)
5. Total good matches: 23
6. Matches are found - 10/10
7. Faces are similar


## input image
Lets change our input image 
![test3](https://user-images.githubusercontent.com/47300390/151328460-76d50a99-0ed3-4f84-a465-867f28e73870.jpg)

## matched regions
![matched_ID_](https://user-images.githubusercontent.com/47300390/151328387-3eed50f5-35d0-4d39-a6ba-a2ea3a5e86a3.png)

## Detected key points and feature comparision from cropped faces
![face_matched](https://user-images.githubusercontent.com/47300390/151328422-c1b092f3-9a11-4536-a8d1-0634c37c0c94.png)

1. Total good matches: 96
2. Rotation Angle: -43 degrees
3. Number of Faces: 1 (input image)
4. Number of Faces: 1 (query image)
5. Total good matches: 0
6. Matches are found - 1/15
7. Faces are not similar
