#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/types.h>
#include <dirent.h>

#include <opencv2/core/core.hpp>

#include "System.h"

using namespace std;
using namespace cv;

void ReadDirectory(const string &name, vector<string> &v);
void LoadBoundingBox(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps);

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
    string path_to_sequence(argv[3]);

    string token;
    stringstream ss(path_to_sequence);
    while (getline(ss, token, '/'));
    cout <<1 << endl;
    vector<string> boundingBoxFilenames;
    cout <<2 << endl;

    ReadDirectory(path_to_sequence + "/" + token + "_CAM_log/", boundingBoxFilenames);
    cout <<3 << endl;

    boundingBoxFilenames.erase(boundingBoxFilenames.begin(), boundingBoxFilenames.begin() + 2);

    VideoCapture cap(path_to_sequence + "/" + token + "_CAM.avi");

    double timestamp;
    int frameId;

    int start = atoi(argv[4]);
    cap.set(CV_CAP_PROP_POS_MSEC, (double)start * 1000);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    // ORB_SLAM2::System SLAM(path_to_vocabulary, path_to_settings, path_to_sequence, ORB_SLAM2::System::MONOCULAR, true);
    ORB_SLAM2::System SLAM(path_to_vocabulary, path_to_settings, path_to_sequence, ORB_SLAM2::System::MONOCULAR, true);
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
            timestamp = cap.get(CV_CAP_PROP_POS_MSEC) / 1000;
            frameId = cap.get(CV_CAP_PROP_POS_FRAMES);

            if (frameId >= int(boundingBoxFilenames.size()))
            {
                cout << "End of file" << endl;
                break;
            }

            // Read image from file
            cap >> im;
        }

        if (im.empty())
        {
            cerr << endl
                 << "Failed to read camera" << endl;
            break;
        }
        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, timestamp, frameId);

        // cout << "timestamp : " << timestamp << " \t frameId : " << frameId << endl;
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void ReadDirectory(const string &name, vector<string> &v)
{
    DIR *dirp = opendir(name.c_str());
    struct dirent *dp;

    while ((dp = readdir(dirp)) != NULL)
    {
        v.push_back(name + dp->d_name);
    }
    cout <<4 << endl;
    closedir(dirp);
}

void LoadBoundingBox(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for (int i = 0; i < nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}