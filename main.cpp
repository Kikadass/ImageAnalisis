
/*
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay( Mat frame );

// Global variables
String face_cascade_name, eyes_cascade_name;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";

// @function main
int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
                             "{help h||}"
                                     "{face_cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
                                     "{eyes_cascade|../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}");

    cout << "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
            "You can use Haar or LBP features.\n\n";
    parser.printMessage();

    face_cascade_name = parser.get<string>("face_cascade");
    eyes_cascade_name = parser.get<string>("eyes_cascade");
    VideoCapture capture;
    Mat frame;

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

    //-- 2. Read the video stream
    capture.open( 0 );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );

        char c = (char)waitKey(10);
        if( c == 27 ) { break; } // escape
    }
    return 0;
}

// @function detectAndDisplay
void detectAndDisplay( Mat frame ){
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
    }
    //-- Show what you got
    imshow( window_name, frame );
}


*/









#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
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

int main( int argc, char** argv ) {
    Mat image;


    // Read the file
    image = imread("/Users/kikepieraserra/Pictures/Navidades 2016-2017/UK/IMG_3859.jpg", CV_LOAD_IMAGE_UNCHANGED);

    if(! image.data ){                              // Check for invalid input
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }


    namedWindow(window_name, CV_WINDOW_AUTOSIZE);// Create a window for display.
    image = scaleDown(image, 4);

    //moveWindow("Display", 700, 700);            // put window in certain position in the screen

    //imwrite("filename.jpg", image); //save image

    //absdiff();   compare 2 images and tell the difference, result: threshold image https://www.youtube.com/watch?v=X6rPdRZzgjg&index=3&list=PLo1wvPF7fMxQ_SXibg1azwBfmTFn02B9O

    //infinite loop
    while (1) {
        // if pressed ESC waits for 1ms
        if (waitKeyEx(1) == 27) {
            return 0;
        }
        //resize(image, image, image.size() / 4, 0.25, 0.25);

    }


}