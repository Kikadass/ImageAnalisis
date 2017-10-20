#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <stdint.h>
#include <opencv/cv.hpp>

using namespace cv;
using namespace std;

String window_name = "Image Analisis";

Mat scaleUp(Mat image, int scale){
    resize(image, image, image.size()*scale, scale, scale);   //resize image
    resizeWindow(window_name, image.cols, image.rows);    // resize window

    imshow(window_name, image);                   // Show our image inside it.

    return image;
}

Mat scaleDown(Mat image, int scale){
    resize(image, image, image.size()/scale, 1/scale, 1/scale);   //resize image
    resizeWindow(window_name, image.cols, image.rows);    // resize window

    imshow(window_name, image);                   // Show our image inside it.

    return image;
}


//same weights
void localNeighborhood(Mat image){
    //modified.at<uint8_t>(r,c) = (original.at<uint8_t>(r, c)) /2; // GRAYSCALE

    for (int r = 0; r < image.rows; r++) {
        for (int c = 0; c < image.cols; c++) {
            int finalRValue = 0;
            int finalGValue = 0;
            int finalBValue = 0;
            int pixelsInMask = 0;

            int neighbourhoodSize = 5;

            for (int i = -neighbourhoodSize/2; i <= neighbourhoodSize/2; i++){
                for (int j = -neighbourhoodSize/2; j <= neighbourhoodSize/2; j++) {
                    if (r+i >= 0 && r+i < image.rows-1 && c+j >= 0 && c+j < image.cols-1){
                        finalRValue += image.at<Vec3b>(r+i,c+j)[2];
                        finalGValue += image.at<Vec3b>(r+i,c+j)[1];
                        finalBValue += image.at<Vec3b>(r+i,c+j)[0];
                        pixelsInMask ++;
                    }
                }
            }

            finalRValue /= pixelsInMask;
            finalGValue /= pixelsInMask;
            finalBValue /= pixelsInMask;

            image.at<Vec3b>(r,c)[0] = finalBValue; //B
            image.at<Vec3b>(r,c)[1] = finalGValue; //G
            image.at<Vec3b>(r,c)[2] = finalRValue; //R

        }
    }
}

//more weight closer to the center
/*  2 ^ (neighbourhoodSize-abs(i)-abs(j));
 * if it was 5x5 mask
 * 2  4  8  4  2
 * 4  8  16 8  4
 * 8  16 32 16 8
 * 4  8  16 8  4
 * 2  4  8  4  2
 *
 * */
void localNeighborhood2(Mat image){
    //modified.at<uint8_t>(r,c) = (original.at<uint8_t>(r, c)) /2; // GRAYSCALE

    for (int r = 0; r < image.rows; r++) {
        for (int c = 0; c < image.cols; c++) {
            int finalRValue = 0;
            int finalGValue = 0;
            int finalBValue = 0;
            int pixelsInMask = 0;

            int neighbourhoodSize = 5;

            for (int i = -neighbourhoodSize/2; i <= neighbourhoodSize/2; i++){
                for (int j = -neighbourhoodSize/2; j <= neighbourhoodSize/2; j++) {
                    if (r+i >= 0 && r+i < image.rows && c+j >= 0 && c+j < image.cols){
                        int multiplier = pow(2, (neighbourhoodSize-abs(i)-abs(j)));
                        //cout << multiplier << endl;

                        finalRValue += image.at<Vec3b>(r + i, c + j)[2]*multiplier;
                        finalGValue += image.at<Vec3b>(r + i, c + j)[1]*multiplier;
                        finalBValue += image.at<Vec3b>(r + i, c + j)[0]*multiplier;
                        pixelsInMask += multiplier;
                    }
                }
            }

            finalRValue /= pixelsInMask;
            finalGValue /= pixelsInMask;
            finalBValue /= pixelsInMask;

            image.at<Vec3b>(r,c)[0] = finalBValue; //B
            image.at<Vec3b>(r,c)[1] = finalGValue; //G
            image.at<Vec3b>(r,c)[2] = finalRValue; //R

        }
    }
}

int getMean(int values [], int arraySize){
    int mean = 0;

    sort(values, values + arraySize);

    if (arraySize%2 == 0){
        mean = (values[arraySize/2-1] + values[arraySize/2]) /2;
    }
    else {
        mean = values[arraySize/2];
    }
    return mean;
}

void edgePreserving(Mat image){
    for (int r = 0; r < image.rows; r++) {
        for (int c = 0; c < image.cols; c++) {
            int neighbourhoodSize = 5;

            int arraySize = neighbourhoodSize*neighbourhoodSize;
            int RValues [arraySize] ;
            int GValues [arraySize] ;
            int BValues [arraySize] ;
            int pixelsInMask = 0;



            for (int i = -neighbourhoodSize/2; i <= neighbourhoodSize/2; i++){
                for (int j = -neighbourhoodSize/2; j <= neighbourhoodSize/2; j++) {
                    if (r+i >= 0 && r+i < image.rows && c+j >= 0 && c+j < image.cols){
                        RValues[pixelsInMask] = image.at<Vec3b>(r + i, c + j)[2];
                        GValues[pixelsInMask] = image.at<Vec3b>(r + i, c + j)[1];
                        BValues[pixelsInMask] = image.at<Vec3b>(r + i, c + j)[0];
                        pixelsInMask++;
                    }
                }
            }

            int finalRValue = getMean(RValues, pixelsInMask);
            int finalGValue = getMean(GValues, pixelsInMask);
            int finalBValue = getMean(BValues, pixelsInMask);

            image.at<Vec3b>(r,c)[0] = finalBValue; //B
            image.at<Vec3b>(r,c)[1] = finalGValue; //G
            image.at<Vec3b>(r,c)[2] = finalRValue; //R

        }
    }
}

int main( int argc, char** argv ) {
    Mat original;
    Mat modified;
    Mat modified2;
    Mat modified3;


    // Read the file
    original = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    modified = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    modified2 = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    modified3 = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_GRAYSCALE);

    if(! original.data ){                              // Check for invalid input
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }


    namedWindow("ORIGINAL", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("MODIFIED", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("MODIFIED2", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("MODIFIED3", CV_WINDOW_AUTOSIZE);// Create a window for display.

    imshow("ORIGINAL", original);
    //imshow("MODIFIED", modified);


    // put each image next to each other
    moveWindow("ORIGINAL", 0, 0);            // put window in certain position in the screen
    moveWindow("MODIFIED", original.cols, 0);            // put window in certain position in the screen
    moveWindow("MODIFIED2", original.cols*2, 0);            // put window in certain position in the screen
    moveWindow("MODIFIED3", original.cols*2, original.rows);            // put window in certain position in the screen

    //imwrite("filename.jpg", image); //save image

    //absdiff(modified, original, modified);   //compare 2 images and tell the difference, result: threshold image https://www.youtube.com/watch?v=X6rPdRZzgjg&index=3&list=PLo1wvPF7fMxQ_SXibg1azwBfmTFn02B9O


    //make images blurry taking random noise away
    localNeighborhood(modified);
    localNeighborhood2(modified2);
    //localNeighborhood2(modified3);
    edgePreserving(modified3);



    // show images
    imshow("ORIGINAL", original);
    imshow("MODIFIED", modified);
    imshow("MODIFIED2", modified2);
    imshow("MODIFIED3", modified3);


    waitKey();
    //infinite loop
    while (1) {
        // if pressed ESC waits for 1ms
        if (waitKeyEx(10) == 27) {
            return 0;
        }
    }
}