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

void negativeImage(Mat& image) {
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            image.at<float>(i, j) = 1 - image.at<float>(i, j);
        }
    }
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

            int neighbourhoodSize = 11;
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


float average_error(Mat* originalImage, Mat* image){
    float Rerror = 0;
    float Gerror = 0;
    float Berror = 0;

    for (int r = 0; r < (*image).rows; r++) {
        for (int c = 0; c < (*image).cols; c++) {

            for (int i = 0; i < 3; i++){
                float tmp = (*originalImage).at<Vec3b>(r,c)[i] - (*image).at<Vec3b>(r,c)[i];

                if (i == 0) Berror += tmp*tmp;
                if (i == 1) Gerror += tmp*tmp;
                if (i == 2) Rerror += tmp*tmp;

            }
        }
    }

    Berror /= (*image).rows*(*image).cols;
    Gerror /= (*image).rows*(*image).cols;
    Rerror /= (*image).rows*(*image).cols;

    return (Rerror+Gerror+Berror)/3;
}

void getVisualDFT(Mat& image, Mat& destImage, bool dftImage){

    Mat planes[2] = { Mat::zeros(image.size(), CV_32F), Mat::zeros(image.size(), CV_32F) };

    //MAGNITUDE
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(image, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magnitudeImage = planes[0];

    magnitudeImage += Scalar::all(1);                    // switch to logarithmic scale

    if (dftImage) {
        log(magnitudeImage, magnitudeImage);  // this log messes up the brightness
    }

    // crop the spectrum, if it has an odd number of rows or columns
    magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));


    normalize(magnitudeImage, magnitudeImage, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    // viewable image form (float between values 0 and 1).

    destImage = magnitudeImage;
}

void showImage(String name, Mat image, bool dftImage){
    Mat imageShow;
    getVisualDFT(image, imageShow, dftImage);
    imshow(name, imageShow);

}

// rearrange the quadrants of Fourier image  so that the origin is at the image center
// its stored again in image
void rearrangeDFT(Mat& image){
    int cx = image.cols/2;
    int cy = image.rows/2;


    Mat q0(image, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(image, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(image, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(image, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void invertDFT(Mat& image, Mat& destImage){
    Mat inverted;
    dft(image, inverted, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    destImage = inverted;
}

void readyDFT(String location, Mat& destImage){
    Mat image = imread(location, CV_LOAD_IMAGE_GRAYSCALE);

    Mat padded;
    //expand input image to optimal size
    // if size is an exponential of 2 it is a lot quicker to process
    int m = getOptimalDFTSize( image.rows );
    int n = getOptimalDFTSize( image.cols );

    // to do that we need to add 0s to the extra pixels
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));


    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};

    Mat complexImage;

    merge(planes, 2, complexImage);         // Add to the expanded another plane with zeros

    dft(complexImage, complexImage);            // this way the result may fit in the source matrix

    destImage = complexImage;

}



void gaussianFilter(Size size, Mat& destImage, int centerX, int centerY, float width, float height, float amp = 1.0f) {
    for (int r = 0; r < size.width; r++) {
        for (int c = 0; c < size.height; c++) {
            float x = (((float)r - centerX) * ((float)r - centerX)) / (2.0f * width * width);
            float y = (((float)c - centerY) * ((float)c - centerY)) / (2.0f * height * height);

            float value = amp * exp(-(x + y));
            destImage.at<float>(r, c) = value;
        }
    }

}

void bandPass(Mat& mask, int centerX, int centerY, float width, float height, int thickness){
    mask = Mat::ones(mask.size(), CV_32F);
    ellipse(mask, Point(centerX, centerY), Size(width, height), 0, 0, 360, Scalar(0, 0, 0), thickness);
}

void lowPass(Mat& mask, int centerX, int centerY, float width, float height){
    mask = Mat::ones(mask.size(), CV_32F);
    ellipse(mask, Point(centerX, centerY), Size(width, height), 0, 0, 360, Scalar(0, 0, 0), -1);
}

void highPass(Mat& mask, int centerX, int centerY, float width, float height){
    lowPass(mask, centerX, centerY, width, height);
    negativeImage(mask);
}

void specialMask(Mat& mask){
    mask = Mat::ones(mask.size(), CV_32F);
    int dy = 43;
    int dx = 77;

    for (int i = -2; i < 3; i++){
        for (int j = -2; j < 3; j++){
            if (i == 0 && j == 0) continue;
            int x = mask.cols/2+dx*i;
            int y = mask.rows/2+dy*j;

            rectangle(mask, Point(x, y), Point(x, y), Scalar(0,0,0), 5);
            //rectangle(mask, Point(mask.cols/2-77*2, mask.rows/2-43), Point(mask.cols/2-77*2, mask.rows/2-43), Scalar(0,0,0), 5);
        }
    }

}

void createMask(Mat& image, int type){
    Mat mask;
    int bSizeY = 30;
    int bSizeX = bSizeY * ((float)image.cols / (float)image.rows);
    int sizex = image.cols;
    int sizey = image.rows;


    mask = Mat(Size(sizex, sizey), CV_32F);

    //gaussianFilter(Size(sizex, sizey), mask, image.cols/2, image.rows/2, bSizeX, bSizeY);
    //bandPass(mask, image.cols/2, image.rows/2, bSizeX, bSizeY, 10);
    //lowPass(mask, image.cols/2, image.rows/2, bSizeX, bSizeY);
    //highPass(mask, image.cols/2, image.rows/2, bSizeX, bSizeY);
    specialMask(mask);

    //normalize(mask, mask, 0, 1, CV_MINMAX);

    showImage("Mask", mask, true);


    // do the same than in the image with the 2 channels
    Mat planes[] = { Mat::zeros(image.size(), CV_32F), Mat::zeros(image.size(), CV_32F) };

    Mat fullMask;
    planes[0] = mask;
    // merge the two planes into the single image
    merge(planes, 2, fullMask);

    // multiply the mask with the image to get the change done
    mulSpectrums(image, fullMask, image, 0);

    //resize(image, image, image.size()*2, 2, 2);
    showImage("Mask added to dft", image, true);
}

Mat createDFT() {
    Mat dftImage;
    readyDFT("../PandaNoise.bmp", dftImage);

    rearrangeDFT(dftImage);

    showImage("spectrum magnitude", dftImage, true);

    createMask(dftImage, 1);

    rearrangeDFT(dftImage);


    Mat inverted;
    invertDFT(dftImage, inverted);

    return inverted;

}

int main( int argc, char** argv ) {
    Mat original;
    Mat noisy;
    Mat modified;
    Mat modified2;
    Mat modified3;
    Mat modified4;
    Mat modified5;
    Mat modified6;


    // Read the file
    original = imread("../PandaOriginal.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    noisy = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_COLOR);
    modified = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_COLOR);
    modified2 = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_COLOR);
    modified3 = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_COLOR);
    modified4 = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_COLOR);
    modified5 = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_COLOR);
    modified6 = imread("../PandaNoise.bmp", CV_LOAD_IMAGE_GRAYSCALE);

    if(! noisy.data ){                              // Check for invalid input
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }


    namedWindow("ORIGINAL", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("NOISY", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("LOCAL NEIGHBOURHOOD", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("GAUSSIAN BLUR", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("EDGE PRESERVING", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("IMAGE SHARPENING", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("EDGE FILTERING", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("DFT", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("Mask added to dft", CV_WINDOW_AUTOSIZE);// Create a window for display.

    imshow("ORIGINAL", original);
    imshow("NOISY", noisy);


    // put each image next to each other
    moveWindow("ORIGINAL", 0, 0);            // put window in certain position in the screen
    moveWindow("NOISY", noisy.cols, 0);           // put window in certain position in the screen
    moveWindow("LOCAL NEIGHBOURHOOD", noisy.cols*2, 0);            // put window in certain position in the screen
    moveWindow("GAUSSIAN BLUR", 0, noisy.rows+50);            // put window in certain position in the screen
    moveWindow("EDGE PRESERVING", noisy.cols, noisy.rows+50);            // put window in certain position in the screen
    moveWindow("IMAGE SHARPENING", noisy.cols*2, noisy.rows+50);            // put window in certain position in the screen
    moveWindow("EDGE FILTERING", 0, (noisy.rows+50)*2);            // put window in certain position in the screen
    moveWindow("DFT", noisy.cols, (noisy.rows+50)*2);            // put window in certain position in the screen
    moveWindow("Mask added to dft", noisy.cols*2, (noisy.rows+50)*2);            // put window in certain position in the screen

    //imwrite("filename.jpg", image); //save image

    //absdiff(modified, noisy, modified);   //compare 2 images and tell the difference, result: threshold image https://www.youtube.com/watch?v=X6rPdRZzgjg&index=3&list=PLo1wvPF7fMxQ_SXibg1azwBfmTFn02B9O


    //make images blurry taking random noise away
    localNeighborhood(&noisy, &modified);
    gaussianBlur(&noisy, &modified2);
    edgePreserving(&noisy, &modified3);
    imageSharpening(&noisy, &modified4);
    edgeFiltering(&modified2, &modified5);
    Mat modified7 = createDFT();

    // show images
    imshow("LOCAL NEIGHBOURHOOD", modified);
    imshow("GAUSSIAN BLUR", modified2);
    imshow("EDGE PRESERVING", modified3);
    imshow("IMAGE SHARPENING", modified4);
    imshow("EDGE FILTERING", modified5);
    showImage("DFT", modified7, false);


    cout << "LOCAL NEIGHBOURHOOD: " << average_error(&original, &modified) << endl;
    cout << "GAUSSIAN BLUR: " << average_error(&original, &modified2) << endl;
    cout << "EDGE PRESERVING: " << average_error(&original, &modified3) << endl;
    cout << "IMAGE SHARPENING: " << average_error(&original, &modified4) << endl;
    cout << "EDGE FILTERING: " << average_error(&original, &modified5) << endl;
    cout << "DFT: " << average_error(&original, &modified6) << endl;

    //infinite loop
    while (1) {
        // if pressed ESC waits for 1ms
        if (waitKeyEx(10) == 27) {
            return 0;
        }
    }
}