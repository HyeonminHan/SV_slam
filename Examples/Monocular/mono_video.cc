#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>

#include<opencv2/core/core.hpp>

#include"System.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        cerr << endl
             << "Usage: ./mono_video path_to_vocabulary path_to_settings path_to_sequence start_point" << endl;
        return 1;
    }
    
    string path_to_vocabulary(argv[1]);
    string path_to_settings(argv[2]);
    VideoCapture cap(argv[3]);

    double delay = 1000 / cap.get(CV_CAP_PROP_FPS);

    int start = atoi(argv[4]);
    cap.set(CV_CAP_PROP_POS_MSEC, (double)start * 1000);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(path_to_vocabulary,path_to_settings,ORB_SLAM2::System::MONOCULAR,true); 

    // Main loop
    cv::Mat im;
    while (true)
    {
        if (SLAM.IsStop())
        {
            continue;
        }
        else
        {
            // Read image from file
            cap >> im;
        }

        if (im.empty())
        {
            cerr << endl
                 << "End of file" << endl;
            break;
        }

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, delay);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");    

    return 0;
}