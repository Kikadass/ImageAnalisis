#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <stdint.h>
#include <opencv/cv.hpp>

using namespace cv;
using namespace std;

String window_name = "Image Analisis";

void scaleUp(Mat* image, int scale){
    resize((*image), (*image), (*image).size()*scale, scale, scale);   //resize image
    resizeWindow(window_name, (*image).cols, (*image).rows);    // resize window

    imshow(window_name, (*image));                   // Show our image inside it.
}

void scaleDown(Mat* image, int scale){
    resize((*image), (*image), (*image).size()/scale, 1/scale, 1/scale);   //resize image
    resizeWindow(window_name, (*image).cols, (*image).rows);    // resize window

    imshow(window_name, (*image));                   // Show our image inside it.
}

double distance(double x1, double y1, double x2, double y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

//same weights
void localNeighborhood(Mat* image, Mat* destImage){
    //modified.at<uint8_t>(r,c) = (original.at<uint8_t>(r, c)) /2; // GRAYSCALE

    for (int r = 0; r < (*image).rows; r++) {
        for (int c = 0; c < (*image).cols; c++) {
            int finalRValue = 0;
            int finalGValue = 0;
            int finalBValue = 0;
            int pixelsInMask = 0;

            int neighbourhoodSize = 5;

            for (int i = -neighbourhoodSize/2; i <= neighbourhoodSize/2; i++){
                for (int j = -neighbourhoodSize/2; j <= neighbourhoodSize/2; j++) {
                    if (r+i >= 0 && r+i < (*image).rows-1 && c+j >= 0 && c+j < (*image).cols-1){
                        finalRValue += (*image).at<Vec3b>(r+i,c+j)[2];
                        finalGValue += (*image).at<Vec3b>(r+i,c+j)[1];
                        finalBValue += (*image).at<Vec3b>(r+i,c+j)[0];
                        pixelsInMask ++;
                    }
                }
            }

            finalRValue /= pixelsInMask;
            finalGValue /= pixelsInMask;
            finalBValue /= pixelsInMask;

            (*destImage).at<Vec3b>(r,c)[0] = finalBValue; //B
            (*destImage).at<Vec3b>(r,c)[1] = finalGValue; //G
            (*destImage).at<Vec3b>(r,c)[2] = finalRValue; //R

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
void gaussianBlur(Mat* image, Mat* destImage){
    cout << "Gaussian blur" << endl;

    //modified.at<uint8_t>(r,c) = (original.at<uint8_t>(r, c)) /2; // GRAYSCALE

    for (int r = 0; r < (*image).rows; r++) {
        for (int c = 0; c < (*image).cols; c++) {
            double finalRValue = 0;
            double finalGValue = 0;
            double finalBValue = 0;
            double pixelsInMask = 0;

            int neighbourhoodSize = 21;
            if (neighbourhoodSize%2 == 0){
                cout << "The neighbourhoodSize needs to be an odd number" << endl;
                throw;
            }
            double max = distance(0, 0, neighbourhoodSize/2, neighbourhoodSize/2);

            for (int i = -neighbourhoodSize/2; i <= neighbourhoodSize/2; i++){
                for (int j = -neighbourhoodSize/2; j <= neighbourhoodSize/2; j++) {
                    if (r+i >= 0 && r+i < (*image).rows && c+j >= 0 && c+j < (*image).cols){
                        //int multiplier = pow(2, (neighbourhoodSize-abs(i)-abs(j)));
                        double multiplier = max - distance(0, 0, i, j);
                        //cout << multiplier << endl;

                        finalRValue += (*image).at<Vec3b>(r + i, c + j)[2]*multiplier;
                        finalGValue += (*image).at<Vec3b>(r + i, c + j)[1]*multiplier;
                        finalBValue += (*image).at<Vec3b>(r + i, c + j)[0]*multiplier;
                        pixelsInMask += multiplier;
                    }
                }
            }

            finalRValue /= pixelsInMask;
            finalGValue /= pixelsInMask;
            finalBValue /= pixelsInMask;

            (*destImage).at<Vec3b>(r,c)[0] = finalBValue; //B
            (*destImage).at<Vec3b>(r,c)[1] = finalGValue; //G
            (*destImage).at<Vec3b>(r,c)[2] = finalRValue; //R

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

void edgePreserving(Mat* image, Mat* destImage){
    for (int r = 0; r < (*image).rows; r++) {
        for (int c = 0; c < (*image).cols; c++) {
            int neighbourhoodSize = 5;

            int arraySize = neighbourhoodSize*neighbourhoodSize;
            int RValues [arraySize] ;
            int GValues [arraySize] ;
            int BValues [arraySize] ;
            int pixelsInMask = 0;



            for (int i = -neighbourhoodSize/2; i <= neighbourhoodSize/2; i++){
                for (int j = -neighbourhoodSize/2; j <= neighbourhoodSize/2; j++) {
                    if (r+i >= 0 && r+i < (*image).rows && c+j >= 0 && c+j < (*image).cols){
                        RValues[pixelsInMask] = (*image).at<Vec3b>(r + i, c + j)[2];
                        GValues[pixelsInMask] = (*image).at<Vec3b>(r + i, c + j)[1];
                        BValues[pixelsInMask] = (*image).at<Vec3b>(r + i, c + j)[0];
                        pixelsInMask++;
                    }
                }
            }

            int finalRValue = getMean(RValues, pixelsInMask);
            int finalGValue = getMean(GValues, pixelsInMask);
            int finalBValue = getMean(BValues, pixelsInMask);

            (*destImage).at<Vec3b>(r,c)[0] = finalBValue; //B
            (*destImage).at<Vec3b>(r,c)[1] = finalGValue; //G
            (*destImage).at<Vec3b>(r,c)[2] = finalRValue; //R

        }
    }
}

// Laplacian edge filtering
/* -1 -1 -1
 * -1  8 -1
 * -1 -1 -1
 * */
void edgeFiltering(Mat* image, Mat* destImage){
    for (int r = 0; r < (*image).rows; r++) {
        for (int c = 0; c < (*image).cols; c++) {
            int neighbourhoodSize = 3;

            int finalRValue = 0;
            int finalGValue = 0;
            int finalBValue = 0;

            for (int i = -neighbourhoodSize/2; i <= neighbourhoodSize/2; i++){
                for (int j = -neighbourhoodSize/2; j <= neighbourhoodSize/2; j++) {
                    if (r+i >= 0 && r+i < (*image).rows && c+j >= 0 && c+j < (*image).cols){

                        int multiplier = -1;
                        if (j == 0 && i == 0){
                            multiplier = 8;
                        }

                        finalRValue += (*image).at<Vec3b>(r + i, c + j)[2]*multiplier;
                        finalGValue += (*image).at<Vec3b>(r + i, c + j)[1]*multiplier;
                        finalBValue += (*image).at<Vec3b>(r + i, c + j)[0]*multiplier;

                    }
                }
            }

            (*destImage).at<Vec3b>(r,c)[0] = finalBValue; //B
            (*destImage).at<Vec3b>(r,c)[1] = finalGValue; //G
            (*destImage).at<Vec3b>(r,c)[2] = finalRValue; //R

        }
    }
}

void imageSharpening(Mat* image, Mat* destImage){
    for (int r = 0; r < (*image).rows; r++) {
        for (int c = 0; c < (*image).cols; c++) {
            int neighbourhoodSize = 2;

            for (int i = 0; i < 3; i++){
                int tmp1;
                int tmp2;
                if (r+neighbourhoodSize < (*image).rows && c+neighbourhoodSize < (*image).cols){
                    tmp1 = abs((*image).at<Vec3b>(r, c)[i] - (*image).at<Vec3b>(r+1, c+1)[i]);
                    tmp2 = abs((*image).at<Vec3b>(r+1, c)[i] - (*image).at<Vec3b>(r, c+1)[i]);
                }
                else if (r+neighbourhoodSize >= (*image).rows && c+neighbourhoodSize < (*image).cols){
                    tmp1 = abs((*image).at<Vec3b>(r, c)[i]);
                    tmp2 = abs(-(*image).at<Vec3b>(r, c+1)[i]);

                }else if (r+neighbourhoodSize < (*image).rows && c+neighbourhoodSize >= (*image).cols) {
                    tmp1 = abs((*image).at<Vec3b>(r, c)[i]);
                    tmp2 = abs((*image).at<Vec3b>(r+1, c)[i]);
                }
                (*destImage).at<Vec3b>(r,c)[i] = tmp1 + tmp2;
            }
        }
    }
}

int main( int argc, char** argv ) {
    Mat original;
    Mat modified;
    Mat modified2;
    Mat modified3;
    Mat modified4;
    Mat modified5;


    // Read the file
    original = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_COLOR);
    modified = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_COLOR);
    modified2 = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_COLOR);
    modified3 = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_COLOR);
    modified4 = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_COLOR);
    modified5 = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_COLOR);

    if(! original.data ){                              // Check for invalid input
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }


    namedWindow("ORIGINAL", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("MODIFIED", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("MODIFIED2", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("EDGE PRESERVING", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("IMAGE SHARPENING", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("EDGE FILTERING", CV_WINDOW_AUTOSIZE);// Create a window for display.

    imshow("ORIGINAL", original);
    //imshow("MODIFIED", modified);


    // put each image next to each other
    moveWindow("ORIGINAL", 0, 0);            // put window in certain position in the screen
    moveWindow("MODIFIED", original.cols, 0);            // put window in certain position in the screen
    moveWindow("MODIFIED2", original.cols*2, 0);            // put window in certain position in the screen
    moveWindow("EDGE PRESERVING", 0, original.rows+50);            // put window in certain position in the screen
    moveWindow("IMAGE SHARPENING", original.cols, original.rows+50);            // put window in certain position in the screen
    moveWindow("EDGE FILTERING", original.cols*2, original.rows+50);            // put window in certain position in the screen

    //imwrite("filename.jpg", image); //save image

    //absdiff(modified, original, modified);   //compare 2 images and tell the difference, result: threshold image https://www.youtube.com/watch?v=X6rPdRZzgjg&index=3&list=PLo1wvPF7fMxQ_SXibg1azwBfmTFn02B9O


    //make images blurry taking random noise away
    localNeighborhood(&original, &modified);
    gaussianBlur(&original, &modified2);
    edgePreserving(&original, &modified3);
    imageSharpening(&original, &modified4);
    edgeFiltering(&original, &modified5);

    // show images
    imshow("ORIGINAL", original);
    imshow("MODIFIED", modified);
    imshow("MODIFIED2", modified2);
    imshow("EDGE PRESERVING", modified3);
    imshow("IMAGE SHARPENING", modified4);
    imshow("EDGE FILTERING", modified5);


    //infinite loop
    while (1) {
        // if pressed ESC waits for 1ms
        if (waitKeyEx(10) == 27) {
            return 0;
        }
    }
}