#include <iostream>
#include "camera/CameraStream.hpp"
#include <opencv2/opencv.hpp>

int main() {

    camera::CameraStream cameraStream;
    cv::VideoCapture cap(cameraStream.init(), cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
        std::cout<<"Failed to open camera."<<std::endl;
        return (-1);
    }

    cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);
    cv::Mat img;

    std::cout << "Hit ESC to exit" << "\n" ;
    while(true)
    {
        if (!cap.read(img)) {
            std::cout<<"Capture read error"<<std::endl;
            break;
        }

        cv::imshow("CSI Camera",img);
        int keycode = cv::waitKey(10) & 0xff ;
        if (keycode == 27) break ;
    }

    cap.release();
    cv::destroyAllWindows() ;
    return EXIT_SUCCESS;
}
