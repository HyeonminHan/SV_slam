/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"
#include "Frame.h"

#include "Optimizer.h"
#include "PnPsolver.h"

#include <iostream>

#include <mutex>

using namespace std;
using namespace cv;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor) : mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
                                                                                                                                                                                              mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys), mpViewer(NULL),
                                                                                                                                                                                              mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
        fps = 30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl
         << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if (DistCoef.rows == 5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if (mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor == System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor == System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    cout << endl
         << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if (sensor == System::STEREO || sensor == System::RGBD)
    {
        mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
        cout << endl
             << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if (sensor == System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if (fabs(mDepthMapFactor) < 1e-5)
            mDepthMapFactor = 1;
        else
            mDepthMapFactor = 1.0f / mDepthMapFactor;
    }
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
        }
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

    mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
    else
        mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

// Frame id + mask
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, const int &frameId)
{
    mImGray = im;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, frameId, mpSystem->mask, mpSystem->strSequenceFile);
    else
        mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, frameId, mpSystem->mask, mpSystem->strSequenceFile);

    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState = mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if (mState == NOT_INITIALIZED)
    {

        MonocularInitialization();

        mpFrameDrawer->Update(this);

        if (mState != OK)
            return;

        mTempFrame = (mCurrentFrame);
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if (!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if (mState == OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if (!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                // bOK = Relocalization();
                mpSystem->Reset();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            if (mState == LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if (!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if (!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint *> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if (!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if (bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if (mbVO)
                        {
                            for (int i = 0; i < mCurrentFrame.N; i++)
                            {
                                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if (bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if (!mbOnlyTracking)
        {
            if (bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if (bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if (bOK)
            mState = OK;
        else
            mState = LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if (bOK)
        {
            // // compute roll & pitch

            ComputeRollPitch();
            // ComputeDistanceVelocity();

            // if (!mpMap->defaultGroundPlane)
            // {
            //     float height = ComputeCameraHeight();
            //     //! dynamic part //
            //     if (mCurrentFrame.mnId % 4 == 0)
            //     {
            //         bool isDynamic = TrackDynamic(height);
            //         // bool isDynamic = TrackDynamic_diff(height);
            //         mPrevFrame = (mTempFrame);
            //         mTempFrame = (mCurrentFrame);
            //         prev_good_BoundingBoxes = cur_good_BoundingBoxes;
            //     }
            // }

            // // prevFrame save
            // if (mCurrentFrame.mnId % 30 == 0)
            // {
            //     mPrevFrame = mCurrentFrame;
            //     prevImg = mImGray;
            // }

            // Update motion model
            if (!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                    if (pMP->Observations() < 1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
            {
                MapPoint *pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if (NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if (mState == LOST)
        {
            if (mpMap->KeyFramesInMap() <= 5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if (!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState == LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState == LOST);
    }
}

void Tracking::MonocularInitialization()
{

    if (!mpInitializer)
    {
        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size() > 100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            if (mpInitializer)
                delete mpInitializer;

            mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if ((int)mCurrentFrame.mvKeys.size() <= 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9, true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

        // Check if there are enough correspondences
        if (nmatches < 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
            return;
        }

        cv::Mat Rcw;                 // Current Camera Rotation
        cv::Mat tcw;                 // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        cout <<"frameId : " <<  mCurrentFrame.mnId << endl;
        bool isshow_camera = false;
        if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated, isshow_camera))
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }

        if (isshow_camera)
        {
            mCurrentFrame.mR = mpInitializer->MyCurrentFrame.mR;
            mCurrentFrame.mt = mpInitializer->MyCurrentFrame.mt;
            cv::Mat n_Tcw = cv::Mat::eye(4, 4, CV_32F);
            mpInitializer->MyCurrentFrame.mR.copyTo(n_Tcw.rowRange(0, 3).colRange(0, 3));
            mpInitializer->MyCurrentFrame.mt.copyTo(n_Tcw.rowRange(0, 3).col(3));

            mpMapDrawer->nI_SetCurrentCameraPose(n_Tcw);
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
    KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
    {
        if (vpAllMapPoints[iMP])
        {
            MapPoint *pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState = OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++)
    {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];

        if (pMP)
        {
            MapPoint *pRep = pMP->GetReplaced();
            if (pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7, true);
    vector<MapPoint *> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame *pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for (int i = 0; i < mLastFrame.N; i++)
    {
        float z = mLastFrame.mvDepth[i];
        if (z > 0)
        {
            vDepthIdx.push_back(make_pair(z, i));
        }
    }

    if (vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(), vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        if (!pMP)
            bCreateNew = true;
        else if (pMP->Observations() < 1)
        {
            bCreateNew = true;
        }

        if (bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

            mLastFrame.mvpMapPoints[i] = pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if (vDepthIdx[j].first > mThDepth && nPoints > 100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

    // Project points seen in previous frame
    int th;
    if (mSensor != System::STEREO)
        th = 15;
    else
        th = 7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

    // If few matches, uses a wider window search
    if (nmatches < 20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);
    }

    if (nmatches < 20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    if (mbOnlyTracking)
    {
        mbVO = nmatchesMap < 10;
        return nmatches > 20;
    }

    return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if (!mbOnlyTracking)
                {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if (mSensor == System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
        return false;

    if (mnMatchesInliers < 30)
        return false;
    else
        return true;
}

bool Tracking::NeedNewKeyFrame()
{
    if (mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose = 0;
    if (mSensor != System::MONOCULAR)
    {
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
            {
                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    // Thresholds
    float thRefRatio = 0.75f;
    if (nKFs < 2)
        thRefRatio = 0.4f;

    if (mSensor == System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

    if ((c1a || c1b || c1c) && c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if (mSensor != System::MONOCULAR)
            {
                if (mpLocalMapper->KeyframesInQueue() < 3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if (!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if (mSensor != System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float, int>> vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(), vDepthIdx.end());

            int nPoints = 0;
            for (size_t j = 0; j < vDepthIdx.size(); j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP)
                    bCreateNew = true;
                else if (pMP->Observations() < 1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                if (bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                    pNewMP->AddObservation(pKF, i);
                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP)
        {
            if (pMP->isBad())
            {
                *vit = static_cast<MapPoint *>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if (pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if (mCurrentFrame.isInFrustum(pMP, 0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if (mSensor == System::RGBD)
            th = 3;
        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            th = 5;
        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        KeyFrame *pKF = *itKF;
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            MapPoint *pMP = *itMP;
            if (!pMP)
                continue;
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
            if (!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame *, int> keyframeCounter;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP->isBad())
            {
                const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i] = NULL;
            }
        }
    }

    if (keyframeCounter.empty())
        return;

    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        KeyFrame *pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max)
        {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80)
            break;

        KeyFrame *pKF = *itKF;

        const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame *> spChilds = pKF->GetChilds();
        for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
        {
            KeyFrame *pChildKF = *sit;
            if (!pChildKF->isBad())
            {
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame *pParent = pKF->GetParent();
        if (pParent)
        {
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
    }

    if (pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75, true);

    vector<PnPsolver *> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint *>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (int i = 0; i < nKFs; i++)
    {
        KeyFrame *pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9, true);

    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nKFs; i++)
        {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver *pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint *> sFound;

                const int np = vbInliers.size();

                for (int j = 0; j < np; j++)
                {
                    if (vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if (nGood < 10)
                    continue;

                for (int io = 0; io < mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if (nGood >= 50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if (!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if (mpViewer)
    {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if (mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer *>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if (mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::ComputeRollPitch()
{
    cv::Mat groundPlane = mpMap->GetGroundPlane();

    if (!groundPlane.empty())
    {
        cv::Mat R = mCurrentFrame.GetRotationInverse();

        cv::Mat zAxis = (cv::Mat_<float>(3, 1) << 0, 0, 1);
        zAxis = R * zAxis;
        float pitch = acos(zAxis.dot(groundPlane.rowRange(0, 3))) * 180.0 / M_PI;
        pitch = 90 - pitch;

        cv::Mat xAxis = (cv::Mat_<float>(3, 1) << 1, 0, 0);
        xAxis = R * xAxis;
        float roll = acos(xAxis.dot(groundPlane.rowRange(0, 3))) * 180.0 / M_PI;
        roll = 90 - roll;

        cout << "track :: roll : " << roll << " / pitch : " << pitch << endl;
    }
}

// void Tracking::ComputeRollPitch(cv::Mat R)
// {

//     cv::Mat R_inv = R.inv();

//     cv::Mat zAxis = (cv::Mat_<float>(3, 1) << 0, 0, 1);
//     zAxis = R_inv * zAxis;
//     float pitch = acos(zAxis.dot(groundPlane.rowRange(0, 3))) * 180.0 / M_PI;
//     pitch = 90 - pitch;

//     cv::Mat xAxis = (cv::Mat_<float>(3, 1) << 1, 0, 0);
//     xAxis = R_inv * xAxis;
//     float roll = acos(xAxis.dot(groundPlane.rowRange(0, 3))) * 180.0 / M_PI;
//     roll = 90 - roll;

//     cout << "track :: roll : " << roll << " / pitch : " << pitch << endl;
// }

void Tracking::ComputeDistanceVelocity()
{
    DetectFeatureInBoundingBox2();
    // TrackDynamic(1);
    // for (int i = 0; i < prev_good_BoundingBoxes.size(); i++)
    // {
    //     for (int j = 0; j < prev_good_BoundingBoxes[i].points.size(); j++)
    //     {
    //         cout << prev_good_BoundingBoxes[i].points[j] << endl;
    //         cout << cur_good_BoundingBoxes[i].points[j] << endl;
    //     }
    //     cout << "----" << endl;
    // }

    vector<cv::Point> cornerPoints;
    Mat prev_img, cur_img;
    mPrevFrame.frameImg.copyTo(prev_img);
    mCurrentFrame.frameImg.copyTo(cur_img);
    cv::cvtColor(prev_img, prev_img, CV_GRAY2BGR);
    cv::cvtColor(cur_img, cur_img, CV_GRAY2BGR);
    if (!prev_img.empty())
    {
        if (prev_good_BoundingBoxes.size() == cur_good_BoundingBoxes.size())
        {
            for (int i = 0; i < prev_good_BoundingBoxes.size(); i++)
            {
                for (int j = 0; j < prev_good_BoundingBoxes[i].points.size(); j++)
                {
                    cv::circle(prev_img, prev_good_BoundingBoxes[i].points[j], 5, cv::Scalar(0, 0, 255), -1);
                    cv::circle(cur_img, cur_good_BoundingBoxes[i].points[j], 5, cv::Scalar(0, 0, 255), -1);
                }
            }
        }
        imshow("prev_img", prev_img);
        imshow("cur_img", cur_img);
    }

    /////////////////save prev frame//////////////////////
    //mPrevFrame = mCurrentFrame;
    //prev_good_BoundingBoxes = cur_good_BoundingBoxes;
}

void Tracking::DetectFeatureInBoundingBox()
{

    // for(int i = 0; i < Current_matchPoints.size() ; i++){
    //     Current_matchPoints[i].clear();
    //     Prev_matchPoints[i].clear();
    //     //vector<Point2f>().swap(Current_matchPoints[i]);
    //     //vector<Point2f>().swap(Prev_matchPoints[i]);
    // }
    Current_matchPoints.clear();
    Prev_matchPoints.clear();
    //vector<vector<Point2f>>().swap(Current_matchPoints);
    //vector<vector<Point2f>>().swap(Prev_matchPoints);

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<FeatureDetector> detector = ORB::create();
    cv::Ptr<DescriptorExtractor> descriptor = ORB::create();

    cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    Current_matchPoints.resize(mCurrentFrame.boundingBoxes.size());
    Prev_matchPoints.resize(mPrevFrame.boundingBoxes.size());
    cv::imshow("mimgray", mImGray);

    for (size_t i = 0; i < mCurrentFrame.boundingBoxes.size(); i++)
    {
        cv::Rect rect(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L, mCurrentFrame.boundingBoxes[i].bbox_T),
                      cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_R, mCurrentFrame.boundingBoxes[i].bbox_B));
        cv::Mat boundingBox = mImGray(rect);

        if (boundingBox.rows > 80)
        {
            vector<cv::Point> cornerPoints;
            cv::goodFeaturesToTrack(boundingBox, cornerPoints, 10, 0.001, 10);
            cv::cvtColor(boundingBox, boundingBox, CV_GRAY2BGR);
            for (int j = 0; j < cornerPoints.size(); j++)
            {
                cv::circle(boundingBox, cornerPoints[j], 2, cv::Scalar(0, 0, 255), -1);
            }
            //cv::imshow("boundingBox" + to_string(mCurrentFrame.boundingBoxes[i].trackingID), boundingBox);
            for (int k = 0; k < mPrevFrame.boundingBoxes.size(); k++)
            {
                if (mCurrentFrame.boundingBoxes[i].trackingID == mPrevFrame.boundingBoxes[k].trackingID)
                {
                    if (!prevImg.empty())
                    {

                        cv::Rect Prevrect(cv::Point2f(mPrevFrame.boundingBoxes[k].bbox_L, mPrevFrame.boundingBoxes[k].bbox_T),
                                          cv::Point2f(mPrevFrame.boundingBoxes[k].bbox_R, mPrevFrame.boundingBoxes[k].bbox_B));

                        cv::Mat PrevboundingBox = prevImg(Prevrect);

                        //cv::Mat a(100, 100, CV_8UC3, cv::Scalar(0, 255, 0));
                        // cv::imshow("boundingBox0",boundingBox);

                        detector->detect(PrevboundingBox, keypoints_1);

                        detector->detect(boundingBox, keypoints_2);
                        descriptor->compute(PrevboundingBox, keypoints_1, descriptors_1);
                        descriptor->compute(boundingBox, keypoints_2, descriptors_2);
                        cv::Mat outimg1;
                        cv::drawKeypoints(PrevboundingBox, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
                        // cv::imshow("ORB matching",outimg1);

                        vector<DMatch> matches;

                        matcher->match(descriptors_1, descriptors_2, matches);

                        double min_dist = 10000, max_dist = 0;

                        for (int j = 0; j < descriptors_1.rows; j++)
                        {
                            double dist = matches[j].distance;
                            if (dist < min_dist)
                                min_dist = dist;
                            if (dist > max_dist)
                                max_dist = dist;
                        }

                        printf("-- Max dist : %f \n", max_dist);
                        printf("-- Min dist : %f \n", min_dist);
                        int cnt = 0;
                        std::vector<DMatch> good_matches;
                        for (int j = 0; j < descriptors_1.rows; j++)
                        {
                            if (matches[j].distance <= max(2 * min_dist, 30.0))
                            {
                                good_matches.push_back(matches[j]);
                                Point2f prev(keypoints_1[matches[j].queryIdx].pt.x + mPrevFrame.boundingBoxes[k].bbox_L,
                                             keypoints_1[matches[j].queryIdx].pt.y + mPrevFrame.boundingBoxes[k].bbox_T);
                                Point2f cur(keypoints_2[matches[j].queryIdx].pt.x + mCurrentFrame.boundingBoxes[i].bbox_L,
                                            keypoints_2[matches[j].queryIdx].pt.y + mCurrentFrame.boundingBoxes[i].bbox_T);

                                Current_matchPoints[i].push_back(cur);
                                Prev_matchPoints[k].push_back(prev);
                                //cout << "prev : " << prev.x << " " << prev.y << endl;
                                //cout << "cur : " << cur.x << " " << cur.y << endl;
                                cnt++;
                            }
                        }
                        cout << "cnt : " << cnt << endl;

                        // Mat curImg, prevImg;
                        // curImg = mImGray;
                        // prevImg = mPrevFrame.frameImg;
                        // cv::cvtColor(curImg, curImg, CV_GRAY2BGR);
                        // cv::cvtColor(prevImg, prevImg, CV_GRAY2BGR);
                        // for(int j = 0; j < Current_matchPoints.size(); j++){
                        //     for(int jj = 0 ; jj< Current_matchPoints[j].size(); jj++){
                        //         cv::circle(curImg, Current_matchPoints[j][jj], 2, cv::Scalar(0, 0, 255), -1);
                        //         cv::circle(prevImg, Prev_matchPoints[j][jj], 2, cv::Scalar(0, 0, 255), -1);
                        //     }
                        // }
                        // imshow("prevImg", prevImg);
                        // imshow("curImg", curImg);

                        cv::Mat img_match;
                        cv::Mat img_goodmatch;
                        cv::drawMatches(PrevboundingBox, keypoints_1, boundingBox, keypoints_2, matches, img_match);
                        cv::drawMatches(PrevboundingBox, keypoints_1, boundingBox, keypoints_2, good_matches, img_goodmatch);

                        // cv::imshow ( "img match", img_match );
                        // cv::imshow("img_goodmatch", img_goodmatch);
                    }
                }
            }
        }
    }

    // for(int i=0; i<Current_matchPoints.size(); i++){
    //     for(int j = 0; j <Current_matchPoints[i].size(); j++){
    //         cout << Current_matchPoints[i][j] << endl;
    //         cout << Prev_matchPoints[i][j] << endl;
    //         cout << " ====" << endl;
    //     }
    // }
}

void Tracking::DetectFeatureInBoundingBox2()
{
    vector<BoundingBox> cur_good_BoundingBoxes_temp;
    vector<BoundingBox> prev_good_BoundingBoxes_temp;

    vector<BoundingBox> new_boundingboxes;

    for (int ci = 0; ci < mCurrentFrame.boundingBoxes.size(); ci++)
    {
        if (mCurrentFrame.boundingBoxes[ci].bbox_B - mCurrentFrame.boundingBoxes[ci].bbox_T < 80)
            continue;

        vector<Point2f> p_prev, p_cur;

        bool is_new_bbox = true;

        for (int pi = 0; pi < prev_good_BoundingBoxes.size(); pi++)
        {
            if (mCurrentFrame.boundingBoxes[ci].trackingID == prev_good_BoundingBoxes[pi].trackingID)
            {

                Mat prev_tracking_img, cur_tracking_img;
                mPrevFrame.frameImg.copyTo(prev_tracking_img, prev_good_BoundingBoxes[pi].mask);
                mCurrentFrame.frameImg.copyTo(cur_tracking_img, mCurrentFrame.boundingBoxes[ci].mask);
                
                vector<uchar> status;
                vector<float> err;
                TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
                p_prev = prev_good_BoundingBoxes[pi].points;
                calcOpticalFlowPyrLK(prev_tracking_img, cur_tracking_img, p_prev, p_cur, status, err, Size(15, 15), 2, criteria);
                is_new_bbox = false;

                if (p_prev.size() > 10)
                {
                    BoundingBox prev_good_bbox, cur_good_bbox;
                    prev_good_bbox = prev_good_BoundingBoxes[pi];
                    cur_good_bbox = mCurrentFrame.boundingBoxes[ci];

                    vector<Point2f> prev_good_points, cur_good_points;
                    int count = 0;
                    Mat temp;
                    mCurrentFrame.frameImg.copyTo(temp);
                    for (int i = 0; i < p_prev.size(); i++)
                    {
                        if(p_cur[i].x < mCurrentFrame.boundingBoxes[ci].bbox_L || p_cur[i].x > mCurrentFrame.boundingBoxes[ci].bbox_R ||
                        p_cur[i].y < mCurrentFrame.boundingBoxes[ci].bbox_T || p_cur[i].y > mCurrentFrame.boundingBoxes[ci].bbox_B )
                            continue;
                        if (status[i] == 1 )
                        {
                            prev_good_points.push_back(p_prev[i]);
                            cur_good_points.push_back(p_cur[i]);
                            count++;
                            cv::circle(temp, p_cur[i], 5, cv::Scalar(0, 0, 255), -1);
                        }
                    }
                    if (count > 10)
                    {
                        prev_good_bbox.points = prev_good_points;
                        cur_good_bbox.points = cur_good_points;
                        prev_good_BoundingBoxes_temp.push_back(prev_good_bbox);
                        cur_good_BoundingBoxes_temp.push_back(cur_good_bbox);
                    }
                }
            }
        }

        if (is_new_bbox)
        {
            if (!mCurrentFrame.boundingBoxes[ci].mask.empty())
            {
                goodFeaturesToTrack(mCurrentFrame.frameImg, p_cur, 100, 0.3, 7, mCurrentFrame.boundingBoxes[ci].mask, 7, false, 0.04);

                if (p_cur.size() > 10)
                {
                    mCurrentFrame.boundingBoxes[ci].points = p_cur;
                    new_boundingboxes.push_back(mCurrentFrame.boundingBoxes[ci]);
                    //cur_good_BoundingBoxes_temp.push_back(mCurrentFrame.boundingBoxes[ci]);

                }
            }
        }
    }

    for(int kk = 0 ; kk < new_boundingboxes.size() ; kk++){
        cur_good_BoundingBoxes_temp.push_back(new_boundingboxes[kk]);
    }

    cur_good_BoundingBoxes = cur_good_BoundingBoxes_temp;
    prev_good_BoundingBoxes = prev_good_BoundingBoxes_temp;

}

bool Tracking::TrackDynamic(float height)
{
    cout << "height: " << height << "prev:: " << mPrevFrame.mnId << "cur:: " << mCurrentFrame.mnId << endl;
    Frame mTempFrame = mPrevFrame;

    vector<vector<cv::Point2d>> track_pt1_set;
    vector<vector<cv::Point2d>> track_pt2_set;
    vector<int> ID_1;
    vector<cv::Point2d> box_Pos_1;
    vector<int> ID_2;
    vector<cv::Point2d> box_Pos_2;
    vector<int> ID_1_inlier;
    vector<int> ID_2_inlier;


    cout << "size:: " << mCurrentFrame.boundingBoxes.size() << " " << mPrevFrame.boundingBoxes.size() << endl;

    int match_size = 0;

    /////////////////TRACKING_POINT////////////////
    for(int i = 0 ;i < prev_good_BoundingBoxes.size(); i ++ )
        cout <<"prev_good number point : " <<  prev_good_BoundingBoxes[i].points.size() << endl;

    for (int i = 0; i < prev_good_BoundingBoxes.size(); i++)
    {
        vector<Point2d> prev_temp, cur_temp;
        ID_1.push_back(cur_good_BoundingBoxes[i].trackingID);
        ID_2.push_back(prev_good_BoundingBoxes[i].trackingID);
        box_Pos_1.push_back(Point2d((double)cur_good_BoundingBoxes[i].bbox_L, (double)cur_good_BoundingBoxes[i].bbox_T));
        box_Pos_2.push_back(Point2d((double)prev_good_BoundingBoxes[i].bbox_L, (double)prev_good_BoundingBoxes[i].bbox_T));

        for (int j = 0; j < prev_good_BoundingBoxes[i].points.size(); j++)
        {
            prev_temp.push_back(Point2d((double)prev_good_BoundingBoxes[i].points[j].x, (double)prev_good_BoundingBoxes[i].points[j].y));
            cur_temp.push_back(Point2d((double)cur_good_BoundingBoxes[i].points[j].x, (double)cur_good_BoundingBoxes[i].points[j].y));
        }
        match_size++;
        track_pt1_set.push_back(cur_temp);
        track_pt2_set.push_back(prev_temp);
    }


    if (track_pt1_set.size() == 0 || track_pt2_set.size() == 0)
    {
        cerr << "No match" << endl;
        return false;
    }

    /////////////////RECT_POINT////////////////

    // track_pt1_set.resize(mCurrentFrame.boundingBoxes.size());
    // track_pt2_set.resize(mPrevFrame.boundingBoxes.size());

    // for (int i = 0; i < track_pt1_set.size(); i++)
    // {
    //     for (int j = 0; j < track_pt2_set.size(); j++)
    //     {
    //         float lenw1_BB = (mCurrentFrame.boundingBoxes[i].bbox_R - mCurrentFrame.boundingBoxes[i].bbox_L) / 2;
    //         float lenh1_BB = (mCurrentFrame.boundingBoxes[i].bbox_B - mCurrentFrame.boundingBoxes[i].bbox_T) / 2;
    //         float lenw2_BB = (mPrevFrame.boundingBoxes[j].bbox_R - mPrevFrame.boundingBoxes[j].bbox_L) / 2;
    //         float lenh2_BB = (mPrevFrame.boundingBoxes[j].bbox_B - mPrevFrame.boundingBoxes[j].bbox_T) / 2;

    //         if (mCurrentFrame.boundingBoxes[i].trackingID == mPrevFrame.boundingBoxes[j].trackingID )
    //         {
    //             ID_1.push_back(mCurrentFrame.boundingBoxes[i].trackingID);
    //             ID_2.push_back(mPrevFrame.boundingBoxes[j].trackingID);
    //             box_Pos_1.push_back(Point2d((double)cur_good_BoundingBoxes[i].bbox_L, (double)cur_good_BoundingBoxes[i].bbox_T));
    //             box_Pos_2.push_back(Point2d((double)prev_good_BoundingBoxes[i].bbox_L, (double)prev_good_BoundingBoxes[i].bbox_T));

    //             // cout << "IDIDIDID: " << mCurrentFrame.boundingBoxes[i].trackingID << " / "<< mPrevFrame.boundingBoxes[j].trackingID << endl;
    //             track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L, mCurrentFrame.boundingBoxes[i].bbox_T));
    //             track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_R, mCurrentFrame.boundingBoxes[i].bbox_T));
    //             track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_R, mCurrentFrame.boundingBoxes[i].bbox_B));
    //             track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L, mCurrentFrame.boundingBoxes[i].bbox_B));

    //             track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L + lenw1_BB, mCurrentFrame.boundingBoxes[i].bbox_T));
    //             track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_R, mCurrentFrame.boundingBoxes[i].bbox_T + lenh1_BB));
    //             track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_R - lenw1_BB, mCurrentFrame.boundingBoxes[i].bbox_B));
    //             track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L, mCurrentFrame.boundingBoxes[i].bbox_B - lenh1_BB));
    //             track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L + lenw1_BB, mCurrentFrame.boundingBoxes[i].bbox_T + lenh1_BB));

    //             track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L, mPrevFrame.boundingBoxes[j].bbox_T));
    //             track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_R, mPrevFrame.boundingBoxes[j].bbox_T));
    //             track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_R, mPrevFrame.boundingBoxes[j].bbox_B));
    //             track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L, mPrevFrame.boundingBoxes[j].bbox_B));
    //             track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L + lenw2_BB, mPrevFrame.boundingBoxes[j].bbox_T));
    //             track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_R, mPrevFrame.boundingBoxes[j].bbox_T + lenh2_BB));
    //             track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_R - lenw2_BB, mPrevFrame.boundingBoxes[j].bbox_B));
    //             track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L, mPrevFrame.boundingBoxes[j].bbox_B - lenh2_BB));
    //             track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L + lenw2_BB, mPrevFrame.boundingBoxes[j].bbox_T + lenh2_BB));

    //             match_size++;
    //         }
    //     }
    // }

    // if (track_pt1_set.size() == 0 || track_pt2_set.size() == 0)
    // {
    //     cerr << "No match" << endl;
    //     return false;
    // }







    ////////////////////////////////////////////
    // cv::Mat src1 = mCurrentFrame.frameImg.clone();
    // cv::cvtColor(src1, src1, cv::COLOR_GRAY2BGR);
    // cv::Mat src2 = mPrevFrame.frameImg.clone();
    // cv::cvtColor(src2, src2, cv::COLOR_GRAY2BGR);
    // cv::Scalar olive(128, 128, 0), violet(221, 160, 221), brown(42, 42, 165);
    // for (int i = 0; i < ID_1.size(); i++)
    // {
    //     string ID_1_str = to_string(int(ID_1[i]));
    //     cv::putText(src1, ID_1_str, Point2f(box_Pos_1[i].x, box_Pos_1[i].y + 60), 0, 2, violet, 2);

    //     for (int k = 0; k < track_pt1_set[i].size(); k++)
    //     {
    //         circle(src1, track_pt1_set[i][k], 8, Scalar(255, 0, 0), -1);
    //     }
    // }


    // for (int i = 0; i < mCurrentFrame.boundingBoxes.size(); i++)
    // {
    //     for (int j = 0; j < mPrevFrame.boundingBoxes.size(); j++)
    //     {
    //         float lenw1_BB = (mCurrentFrame.boundingBoxes[i].bbox_R - mCurrentFrame.boundingBoxes[i].bbox_L) / 2;
    //         float lenh1_BB = (mCurrentFrame.boundingBoxes[i].bbox_B - mCurrentFrame.boundingBoxes[i].bbox_T) / 2;
    //         float lenw2_BB = (mPrevFrame.boundingBoxes[j].bbox_R - mPrevFrame.boundingBoxes[j].bbox_L) / 2;
    //         float lenh2_BB = (mPrevFrame.boundingBoxes[j].bbox_B - mPrevFrame.boundingBoxes[j].bbox_T) / 2;

    //         // float diff_volume = abs((lenw1_BB * 2 * lenh1_BB * 2) - (lenw2_BB * 2 * lenh2_BB * 2));
    //         int k_idx = 0;
    //         if (mCurrentFrame.boundingBoxes[i].trackingID == mPrevFrame.boundingBoxes[j].trackingID)
    //         {
    //             cv::Rect rect1(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L, mCurrentFrame.boundingBoxes[i].bbox_T), cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_R, mCurrentFrame.boundingBoxes[i].bbox_B));
    //             cv::rectangle(src1, rect1, cv::Scalar(0, 0, 255), 2);
    //             cv::Rect rect2(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L, mPrevFrame.boundingBoxes[j].bbox_T), cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_R, mPrevFrame.boundingBoxes[j].bbox_B));
    //             cv::rectangle(src2, rect2, cv::Scalar(0, 0, 255), 2);
    //         }
    //         else
    //             continue;
    //     }
    // }
    // cv::resize(src1, src1, cv::Size(src1.cols / 3, src1.rows / 3), 0, 0, CV_INTER_NN);
    // imshow("cur", src1);
    // //cvWaitKey(1);
    ///////////////////////////////////////////////////////////////////////////////////

    cout << "matches size: " << match_size << endl;
    float i_width = mCurrentFrame.frameImg.cols;
    float i_height = mCurrentFrame.frameImg.rows;
    cout << i_width << "  ,  " << i_height << endl;

    cv::Mat Rcw1 = mCurrentFrame.mRcw;
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mCurrentFrame.mtcw;
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0, 3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Tcw1_44 = cv::Mat::eye(4, 4, CV_32F);
    Rcw1.copyTo(Tcw1_44.rowRange(0, 3).colRange(0, 3));
    tcw1.copyTo(Tcw1_44.rowRange(0, 3).col(3));

    cv::Mat Ow1 = mCurrentFrame.mOw;
    cv::Mat Twc1 = cv::Mat::eye(4, 4, CV_32F);
    Rwc1.copyTo(Twc1.rowRange(0, 3).colRange(0, 3));
    Ow1.copyTo(Twc1.rowRange(0, 3).col(3));

    cv::Mat Rcw2 = mPrevFrame.mRcw;
    cv::Mat Rwc2 = Rcw2.t();
    cv::Mat tcw2 = mPrevFrame.mtcw;
    cv::Mat Tcw2(3, 4, CV_32F);
    Rcw2.copyTo(Tcw2.colRange(0, 3));
    tcw2.copyTo(Tcw2.col(3));
    cv::Mat Tcw2_44 = cv::Mat::eye(4, 4, CV_32F);
    Rcw2.copyTo(Tcw2_44.rowRange(0, 3).colRange(0, 3));
    tcw2.copyTo(Tcw2_44.rowRange(0, 3).col(3));

    cv::Mat Ow2 = mPrevFrame.mOw;
    cv::Mat Twc2 = cv::Mat::eye(4, 4, CV_32F);
    Rwc2.copyTo(Twc2.rowRange(0, 3).colRange(0, 3));
    Ow2.copyTo(Twc2.rowRange(0, 3).col(3));

    cv::Mat T12_static = Tcw1 * Twc2;
    cv::Mat R12_static = T12_static.rowRange(0, 3).colRange(0, 3);
    cv::Mat t12_static = T12_static.rowRange(0, 3).col(3);

    cv::Mat R21_static = R12_static.t();
    cv::Mat t21_static = -R12_static.t() * t12_static;
    cv::Mat T21_static = cv::Mat::eye(4, 4, CV_32F);
    R21_static.copyTo(T21_static.rowRange(0, 3).colRange(0, 3));
    t21_static.copyTo(T21_static.rowRange(0, 3).col(3));

    // cout << "Twc2: " << Twc2.rows << " " << Twc2.cols << Tcw2<< endl;
    // cout << "T12_static: " << T12_static.rows << " " << T12_static.cols << T12_static<< endl;
    // cout << "R12_static*Twc2: " << R12_static.rows << " " << R12_static.cols << R12_static*Tcw2_44 << endl;
    // cout << "Tcw1: " << Tcw1.rows << " " << Tcw1.cols << Tcw1<< endl;
    // cout << "Tcw1_44: " << Tcw1_44.rows << " " << Tcw1_44.cols << Tcw1_44<< endl;

    vector<float> BB_distance(match_size, -1);
    vector<cv::Point2d> track_pt1_inlier;
    vector<cv::Point2d> track_pt2_inlier;

    for (int i = 0; i < match_size; i++)
    {
        ///tracked feature <- input
        vector<cv::Point2d> track_pt1 = track_pt1_set[i]; //Cur
        vector<cv::Point2d> track_pt2 = track_pt2_set[i]; //Last


        vector<cv::Point2d> track_pt1_inlier_Temp(track_pt1.size());
        vector<cv::Point2d> track_pt2_inlier_Temp(track_pt2.size());

        if (track_pt1.size() < 8 || track_pt2.size() < 8)
        {
            cerr << "point size small,,,TT" << endl;
            continue;
        }
        vector<uchar> inliers;
        cv::Mat dF = findFundamentalMat(track_pt2, track_pt1, cv::RANSAC, 3, 0.99, inliers);

        if (dF.empty())
            continue;
        // cout << " F: " << dF.rows << " " << dF.cols << "\n" << dF << endl;

        cv::Mat F;
        dF.convertTo(F, CV_32F);
        cv::Mat dE = mCurrentFrame.mK.t() * F * mCurrentFrame.mK;
        cv::Mat E;
        dE.convertTo(E, CV_64F);

        ///get RT -> output
        cv::Point2d dpp = cv::Point2d(mCurrentFrame.cx, mCurrentFrame.cy); //pp
        double focal_x = mCurrentFrame.fx;                                 //f
        double focal_y = mCurrentFrame.fy;

        for (int k = 0; k < track_pt1.size(); k++)
        {
            float a = track_pt1[k].x * dE.at<float>(0, 0) + track_pt1[k].y * dE.at<float>(1, 0) + dE.at<float>(2, 0);
            float b = track_pt1[k].x * dE.at<float>(0, 1) + track_pt1[k].y * dE.at<float>(1, 1) + dE.at<float>(2, 1);
            float c = track_pt1[k].x * dE.at<float>(0, 2) + track_pt1[k].y * dE.at<float>(1, 2) + dE.at<float>(2, 2);

            float num = a * track_pt2[k].x + b * track_pt2[k].y + c;
            float den = a * a + b * b;

            if (den == 0)
                continue;

            float dsqr = num * num / den;
                cout << "dsqr: " << dsqr << endl;
            if (1)
            // if (dsqr < 3.84 * 2)
            {
                ID_1_inlier.push_back(ID_1[k]);
                ID_2_inlier.push_back(ID_2[k]);
                track_pt1_inlier_Temp[k].x = track_pt1[k].x;
                track_pt1_inlier_Temp[k].y = track_pt1[k].y;
                track_pt2_inlier_Temp[k].x = track_pt2[k].x;
                track_pt2_inlier_Temp[k].y = track_pt2[k].y;
            }
            else
            {

                track_pt1_inlier_Temp[k].x = -1;
                track_pt1_inlier_Temp[k].y = -1;
                track_pt2_inlier_Temp[k].x = -1;
                track_pt2_inlier_Temp[k].y = -1;
            }
        }

        cv::Mat dR;
        cv::Mat dt;
        for (int k = 0; k < track_pt1_inlier_Temp.size(); k++)
        {
            if (track_pt1_inlier_Temp[k].x != -1 || track_pt1_inlier_Temp[k].y != -1 || track_pt2_inlier_Temp[k].x != -1 || track_pt2_inlier_Temp[k].y != -1)
            {
                track_pt1_inlier.push_back(track_pt1_inlier_Temp[k]);
                track_pt2_inlier.push_back(track_pt2_inlier_Temp[k]);
            }
        }

        if (track_pt1_inlier.size() < 5 || track_pt2_inlier.size() < 5)
        {
            cerr << "point size small for E,,,TT " << track_pt1_inlier.size() << " " << track_pt2_inlier.size() << endl;
            return false;
        }

        if(cv::recoverPose(E,
                        track_pt2_inlier,
                        track_pt1_inlier,
                        dR, dt, focal_x,
                        dpp)==0) continue;
        if (dR.empty() || dt.empty())
        {
            continue;
        } //? scale ambiguity

        ///calculate 3D data -> output
        cv::Mat R00 = cv::Mat::eye(4, 4, CV_64F);
        cv::Mat R01 = cv::Mat::eye(4, 4, CV_64F);
        dR.copyTo(R01.rowRange(0, 3).colRange(0, 3));
        dt.copyTo(R01.rowRange(0, 3).col(3));
        // cout << "R00 : " << R00.rows << " " << R00.cols << R00 << endl;
        // cout << "R01 : " << R01.rows << " " << R01.cols << R01 << endl;

        cv::Mat R12_dy = cv::Mat::eye(4, 4, CV_32F);
        R01.convertTo(R12_dy, CV_32F);
        cv::Mat T_diff;
        T_diff = R12_dy * T21_static;

        cv::Mat R_diff = T_diff.rowRange(0, 3).colRange(0, 3);
        cv::Mat t_diff = T_diff.rowRange(0, 3).col(3);
        cv::Mat R_diff_inv = R_diff.t();
        cv::Mat t_diff_inv = -R_diff.t() * t_diff;
        cv::Mat T_diff_inv = cv::Mat::eye(4, 4, CV_32F);
        R_diff_inv.copyTo(T_diff_inv.rowRange(0, 3).colRange(0, 3));
        t_diff_inv.copyTo(T_diff_inv.rowRange(0, 3).col(3));

        // cout << "T_diff: " << T_diff.rows << " " << T_diff.cols << T_diff<<endl;
        // cout << "T12_static: " << T12_static.rows << " " << T12_static.cols<< T12_static<<endl;
        // cout << "R01: " << R01.rows << " " << R01.cols<< R01<<endl;
        // cout << "R12_dy: " << R12_dy.rows << " " << R12_dy.cols << R12_dy<<endl;

        cv::Mat Ow1_C = Rcw1 * Ow1 + tcw1;
        // cout << "Ow1: " << Ow1.rows << " " << Ow1.cols << Ow1<<endl;
        // cout << "Rcw1: " << Rcw1.rows << " " << Rcw1.cols << Rcw1<<endl;
        // cout << "tcw1: " << tcw1.rows << " " << tcw1.cols << tcw1<<endl;
        // cout << "Ow1_C: " << Ow1_C.rows << " " << Ow1_C.cols << Ow1_C<<endl;

        std::vector<float> dist_vec(track_pt1_inlier.size(), -1);
        for (int k = 0; k < track_pt1_inlier.size(); k++)
        {
            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (track_pt1_inlier[k].x - dpp.x) * (1 / focal_x), (track_pt1_inlier[k].y - dpp.y) * (1 / focal_y), 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (track_pt2_inlier[k].x - dpp.x) * (1 / focal_x), (track_pt2_inlier[k].y - dpp.y) * (1 / focal_y), 1.0);

            //ray dynamic2static img
            cv::Mat xns = R_diff_inv * xn1 + t_diff_inv;
            // cv::Mat xns = (cv::Mat_<float>(3, 1) << (xn1.at<float>(0) + xn2.at<float>(0)) / 2, (xn2.at<float>(1) + xn2.at<float>(1)) / 2, 1.0);
            // Euclidean coordinates
            xns = xns.rowRange(0, 3) / xns.at<float>(2);

            cv::Mat xn1_img = mCurrentFrame.mK * xn1;
            cv::Mat xn2_img = mCurrentFrame.mK * xn2;
            cv::Mat xns_img = mCurrentFrame.mK * xns;

            if (xns_img.at<float>(0) > i_width || xns_img.at<float>(0) < 0 || xns_img.at<float>(1) > i_height || xns_img.at<float>(1) < 0)
                continue;

            // cout << "xn1: " << xn1_img.rows << " " << xn1_img.cols << xn1_img<<endl;
            // cout << "xn2: " << xn2_img.rows << " " << xn2_img.cols << xn2_img<<endl;
            // cout << "xns: " << xns_img.rows << " " << xns_img.cols << xns_img<<endl;
            // cout << "t_diff_inv: " << t_diff_inv.rows << " " << t_diff_inv.cols << t_diff_inv << endl;
            // cout << "==== " << endl;

            cv::Mat ray1 = Rwc1 * xns;
            cv::Mat ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));
            // cout << "cosParallaxRays:: " << cosParallaxRays << endl;

            cv::Mat x3D;
            cv::Mat x3D_C;
            if (cosParallaxRays < 0.9998)
            {
                // Linear Triangulation Method
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = xns.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                A.row(1) = xns.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                cv::Mat w, u, vt;
                cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if (x3D.at<float>(3) == 0)
                    continue;

                // Euclidean coordinates

                x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
                x3D = x3D / height * 1.69;
                float distance = sqrt(pow(x3D.at<float>(0) - Ow1.at<float>(0, 0), 2) + pow(x3D.at<float>(1) - Ow1.at<float>(1, 0), 2) + pow(x3D.at<float>(2) - Ow1.at<float>(2, 0), 2));

                // x3D_C = Rcw1 * x3D + tcw1;
                // x3D_C = x3D_C / height * 1.69;
                // float distance = sqrt(pow(x3D_C.at<float>(0) , 2) + pow(x3D_C.at<float>(1), 2) + pow(x3D_C.at<float>(2) , 2));

                // cout << x3D << " vs " << Ow1 << endl;
                // cout << "x3D: " << x3D.rows << " " << x3D.cols << x3D << endl;
                // cout << "x3D_C: " << x3D_C.rows << " " << x3D_C.cols << x3D_C << endl;

                dist_vec[k] = (distance - 1);
            }
            else
            {
                continue; //No stereo and very low parallax
            }
        }

        int cnt_avg = 0;
        float avg_dist = 0;
        for (int k = 0; k < dist_vec.size(); k++)
        {
            if (dist_vec[k] != -1)
            {
                avg_dist += dist_vec[k];
                cnt_avg++;
            }
        }
        avg_dist /= cnt_avg;

        if (cnt_avg)
        {
            BB_distance[i] = (avg_dist);
        }
        else
            continue;
    }

    cout << "BB_distance.size(): " << BB_distance.size() << endl;

    for (int i = 0; i < BB_distance.size(); i++)
    {
        cout << "ID: " << ID_1_inlier[i] << " dist : " << BB_distance[i] << endl;
    }

    cv::Mat src1 = mCurrentFrame.frameImg.clone();
    cv::cvtColor(src1, src1, cv::COLOR_GRAY2BGR);
    cv::Mat src2 = mPrevFrame.frameImg.clone();
    cv::cvtColor(src2, src2, cv::COLOR_GRAY2BGR);
    cout << 1 << endl;
    cv::Scalar olive(128, 128, 0), violet(221, 160, 221), brown(42, 42, 165);
    for (int i = 0; i < ID_1_inlier.size(); i++)
    {
        string ID_1_str = to_string(int(ID_1_inlier[i]));
        //cv::putText(src1, ID_1_str, Point2f(box_Pos_1[i].x, box_Pos_1[i].y + 60), 0, 2, violet, 2);

        string carDist = to_string(int(BB_distance[i]));
        cv::putText(src1, carDist, Point2f(track_pt1_inlier[i].x, track_pt1_inlier[i].y), 0, 2, cv::Scalar(0, 0, 255), 2);

        for (int k = 0; k < track_pt1_set[i].size(); k++)
        {
            circle(src1, track_pt1_set[i][k], 8, Scalar(255, 0, 0), -1);
        }
    }


    for (int i = 0; i < mCurrentFrame.boundingBoxes.size(); i++)
    {
        for (int j = 0; j < mPrevFrame.boundingBoxes.size(); j++)
        {
            float lenw1_BB = (mCurrentFrame.boundingBoxes[i].bbox_R - mCurrentFrame.boundingBoxes[i].bbox_L) / 2;
            float lenh1_BB = (mCurrentFrame.boundingBoxes[i].bbox_B - mCurrentFrame.boundingBoxes[i].bbox_T) / 2;
            float lenw2_BB = (mPrevFrame.boundingBoxes[j].bbox_R - mPrevFrame.boundingBoxes[j].bbox_L) / 2;
            float lenh2_BB = (mPrevFrame.boundingBoxes[j].bbox_B - mPrevFrame.boundingBoxes[j].bbox_T) / 2;

            // float diff_volume = abs((lenw1_BB * 2 * lenh1_BB * 2) - (lenw2_BB * 2 * lenh2_BB * 2));
            int k_idx = 0;
            if (mCurrentFrame.boundingBoxes[i].trackingID == mPrevFrame.boundingBoxes[j].trackingID)
            {
                cv::Rect rect1(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L, mCurrentFrame.boundingBoxes[i].bbox_T), cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_R, mCurrentFrame.boundingBoxes[i].bbox_B));
                cv::rectangle(src1, rect1, cv::Scalar(0, 0, 255), 2);
                cv::Rect rect2(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L, mPrevFrame.boundingBoxes[j].bbox_T), cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_R, mPrevFrame.boundingBoxes[j].bbox_B));
                cv::rectangle(src2, rect2, cv::Scalar(0, 0, 255), 2);
            }
            else
                continue;
        }
    }
    //cv::resize(src1, src1, cv::Size(src1.cols / 3, src1.rows / 3), 0, 0, CV_INTER_NN);
    imshow("cur", src1);
    // cv::resize(src2, src2, cv::Size(src2.cols / 3, src2.rows / 3), 0, 0, CV_INTER_NN);
    // imshow("prev", src2);
    //cvWaitKey(1);
}

bool Tracking::TrackDynamic_diff(float height)
{
    cout << "height: " << height << "prev:: " << mPrevFrame.mnId << "cur:: " << mCurrentFrame.mnId << endl;
    Frame mTempFrame = mPrevFrame;

    vector<vector<cv::Point2d>> track_pt1_set;
    vector<vector<cv::Point2d>> track_pt2_set;
    vector<int> ID_1;
    vector<cv::Point2d> box_Pos_1;
    vector<int> ID_2;
    vector<cv::Point2d> box_Pos_2;

    // cout << "mnid:: " << mCurrentFrame.mnId << endl;
    cout << "size:: " << mCurrentFrame.boundingBoxes.size() << " " << mPrevFrame.boundingBoxes.size() << endl;

    int match_size = 0;

    /////////////////TRACKING_POINT////////////////

    // for (int i = 0; i < prev_good_BoundingBoxes.size(); i++)
    // {
    //     vector<Point2d> prev_temp, cur_temp;
    //     ID_1.push_back(cur_good_BoundingBoxes[i].trackingID);
    //     ID_2.push_back(prev_good_BoundingBoxes[i].trackingID);
    //     box_Pos_1.push_back(Point2d((double)cur_good_BoundingBoxes[i].bbox_L, (double)cur_good_BoundingBoxes[i].bbox_T));
    //     box_Pos_2.push_back(Point2d((double)prev_good_BoundingBoxes[i].bbox_L, (double)prev_good_BoundingBoxes[i].bbox_T));

    //     for (int j = 0; j < prev_good_BoundingBoxes[i].points.size(); j++)
    //     {
    //         prev_temp.push_back(Point2d((double)prev_good_BoundingBoxes[i].points[j].x, (double)prev_good_BoundingBoxes[i].points[j].y));
    //         cur_temp.push_back(Point2d((double)cur_good_BoundingBoxes[i].points[j].x, (double)cur_good_BoundingBoxes[i].points[j].y));
    //     }
    //     match_size++;
    //     track_pt1_set.push_back(cur_temp);
    //     track_pt2_set.push_back(prev_temp);
    // }
    // if (track_pt1_set.size() == 0 || track_pt2_set.size() == 0)
    // {
    //     cerr << "No match" << endl;
    //     return false;
    // }

    /////////////////RECT_POINT////////////////

    track_pt1_set.resize(mCurrentFrame.boundingBoxes.size());
    track_pt2_set.resize(mPrevFrame.boundingBoxes.size());

    for (int i = 0; i < track_pt1_set.size(); i++)
    {
        for (int j = 0; j < track_pt2_set.size(); j++)
        {
            float lenw1_BB = (mCurrentFrame.boundingBoxes[i].bbox_R - mCurrentFrame.boundingBoxes[i].bbox_L) / 2;
            float lenh1_BB = (mCurrentFrame.boundingBoxes[i].bbox_B - mCurrentFrame.boundingBoxes[i].bbox_T) / 2;
            float lenw2_BB = (mPrevFrame.boundingBoxes[j].bbox_R - mPrevFrame.boundingBoxes[j].bbox_L) / 2;
            float lenh2_BB = (mPrevFrame.boundingBoxes[j].bbox_B - mPrevFrame.boundingBoxes[j].bbox_T) / 2;

            float diff_volume = abs((lenw1_BB * 2 * lenh1_BB * 2) - (lenw2_BB * 2 * lenh2_BB * 2));

            if (mCurrentFrame.boundingBoxes[i].trackingID == mPrevFrame.boundingBoxes[j].trackingID && diff_volume < 10000)
            {
                ID_1.push_back(mCurrentFrame.boundingBoxes[i].trackingID);
                ID_2.push_back(mPrevFrame.boundingBoxes[j].trackingID);
                box_Pos_1.push_back(Point2d((double)cur_good_BoundingBoxes[i].bbox_L, (double)cur_good_BoundingBoxes[i].bbox_T));
                box_Pos_2.push_back(Point2d((double)prev_good_BoundingBoxes[i].bbox_L, (double)prev_good_BoundingBoxes[i].bbox_T));

                // cout << "IDIDIDID: " << mCurrentFrame.boundingBoxes[i].trackingID << " / "<< mPrevFrame.boundingBoxes[j].trackingID << endl;
                track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L, mCurrentFrame.boundingBoxes[i].bbox_T));
                track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_R, mCurrentFrame.boundingBoxes[i].bbox_T));
                track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_R, mCurrentFrame.boundingBoxes[i].bbox_B));
                track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L, mCurrentFrame.boundingBoxes[i].bbox_B));

                track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L + lenw1_BB, mCurrentFrame.boundingBoxes[i].bbox_T));
                track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_R, mCurrentFrame.boundingBoxes[i].bbox_T + lenh1_BB));
                track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_R - lenw1_BB, mCurrentFrame.boundingBoxes[i].bbox_B));
                track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L, mCurrentFrame.boundingBoxes[i].bbox_B - lenh1_BB));
                track_pt1_set[match_size].push_back(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L + lenw1_BB, mCurrentFrame.boundingBoxes[i].bbox_T + lenh1_BB));

                track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L, mPrevFrame.boundingBoxes[j].bbox_T));
                track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_R, mPrevFrame.boundingBoxes[j].bbox_T));
                track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_R, mPrevFrame.boundingBoxes[j].bbox_B));
                track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L, mPrevFrame.boundingBoxes[j].bbox_B));
                track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L + lenw2_BB, mPrevFrame.boundingBoxes[j].bbox_T));
                track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_R, mPrevFrame.boundingBoxes[j].bbox_T + lenh2_BB));
                track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_R - lenw2_BB, mPrevFrame.boundingBoxes[j].bbox_B));
                track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L, mPrevFrame.boundingBoxes[j].bbox_B - lenh2_BB));
                track_pt2_set[match_size].push_back(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L + lenw2_BB, mPrevFrame.boundingBoxes[j].bbox_T + lenh2_BB));

                match_size++;
            }
        }
    }

    if (track_pt1_set.size() == 0 || track_pt2_set.size() == 0)
    {
        cerr << "No match" << endl;
        return false;
    }
    cout << "matches size: " << match_size << endl;
    // for (int i = 0; i < match_size ; i++)
    // {
    // cout <<ID_1[i] << " - " << ID_2[i] << endl;

    // for (int j = 0; j < track_pt1_set[i].size() ; j++)
    // {
    // cout << j << ": " << track_pt1_set[i][j].x << " " << track_pt1_set[i][j].y << endl;
    // cout << j << ": " << track_pt2_set[i][j].x << " " << track_pt2_set[i][j].y << endl << endl;
    // }
    // cout << endl << endl << endl;;
    // }

    cv::Mat Rcw1 = mCurrentFrame.mRcw;
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mCurrentFrame.mtcw;
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0, 3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mCurrentFrame.mOw;
    cv::Mat Twc1 = cv::Mat::eye(4, 4, CV_32F);
    Rwc1.copyTo(Twc1.rowRange(0, 3).colRange(0, 3));
    Ow1.copyTo(Twc1.rowRange(0, 3).col(3));

    cv::Mat Rcw2 = mTempFrame.mRcw;
    cv::Mat Rwc2 = Rcw2.t();
    cv::Mat tcw2 = mTempFrame.mtcw;
    cv::Mat Tcw2(3, 4, CV_32F);
    Rcw2.copyTo(Tcw2.colRange(0, 3));
    tcw2.copyTo(Tcw2.col(3));
    cv::Mat Ow2 = mTempFrame.mOw;
    cv::Mat Twc2 = cv::Mat::eye(4, 4, CV_32F);
    Rwc2.copyTo(Twc2.rowRange(0, 3).colRange(0, 3));
    Ow2.copyTo(Twc2.rowRange(0, 3).col(3));

    cv::Mat R12 = Tcw1 * Twc2;

    // cout << "Rwc2: " << Rwc2.rows << " " << Rwc2.cols << Rwc2<< endl;
    // cout << "Ow2: " << Ow2.rows << " " << Ow2.cols << Ow2<< endl;
    // cout << "Twc: " << Twc2.rows << " " << Twc2.cols << Twc2<< endl;

    vector<float> BB_distance(match_size, -1);
    for (int i = 0; i < match_size; i++)
    {
        ///tracked feature <- input
        vector<cv::Point2d> track_pt1 = track_pt1_set[i]; //Cur
        vector<cv::Point2d> track_pt2 = track_pt2_set[i]; //Last

        if (track_pt1.size() < 8 || track_pt2.size() < 8)
        {
            cerr << "point size small,,,TT" << endl;
            return false;
        }

        vector<uchar> inliers;
        cv::Mat dF = findFundamentalMat(track_pt2, track_pt1, cv::RANSAC, 3, 0.99, inliers);
        if (dF.empty())
        {
            continue;
        }
        // cout << " F: " << dF.rows << " " << dF.cols << "\n" << dF << endl;

        cv::Mat F;
        dF.convertTo(F, CV_32F);
        cv::Mat dE = mCurrentFrame.mK.t() * F * mCurrentFrame.mK;
        cv::Mat E;
        dE.convertTo(E, CV_64F);

        ///get RT -> output
        cv::Point2d dpp = cv::Point2d(mCurrentFrame.cx, mCurrentFrame.cy); //pp
        double focal_x = mCurrentFrame.fx;                                 //f
        double focal_y = mCurrentFrame.fy;

        Mat essential_matrix = findEssentialMat(track_pt2,
                                                track_pt1,
                                                focal_x,
                                                dpp,
                                                cv::RANSAC,
                                                0.99,
                                                2.0);
        cout << essential_matrix.size() << endl;
        cout << essential_matrix << endl;

        cv::Mat dR;
        cv::Mat dt;
        cv::recoverPose(essential_matrix,
                        track_pt2,
                        track_pt1,
                        dR, dt, focal_x,
                        dpp);
        if (dR.empty() || dt.empty())
        {
            continue;
        }

        ///calculate 3D data -> output
        cv::Mat R00 = cv::Mat::eye(4, 4, CV_64F);
        cv::Mat R01 = cv::Mat::eye(4, 4, CV_64F);
        dR.copyTo(R01.rowRange(0, 3).colRange(0, 3));
        dt.copyTo(R01.rowRange(0, 3).col(3));

        // cout << "R00 : " << R00.rows << " " << R00.cols << R00 << endl;
        // cout << "R01 : " << R01.rows << " " << R01.cols << R01 << endl;

        std::vector<float> dist_vec(track_pt1.size(), -1);
        for (int k = 0; k < track_pt1.size(); k++)
        {
            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (track_pt1[k].x - dpp.x) * (1 / focal_x), (track_pt1[k].y - dpp.y) * (1 / focal_y), 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (track_pt2[k].x - dpp.x) * (1 / focal_x), (track_pt2[k].y - dpp.y) * (1 / focal_y), 1.0);

            // cout << "xn1: " << xn1 << " " << track_pt1[k].x << " " << dpp.x << " " << mCurrentFrame.invfx << endl;
            // cout << "xn2: " << xn2 << " " << track_pt2[k].x << " " << dpp.x << " " << mCurrentFrame.invfy << endl;

            cv::Mat ray1 = Rwc1 * xn1;
            cv::Mat ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));
            // cout << "cosParallaxRays:: " << cosParallaxRays << endl;

            cv::Mat x3D;
            cv::Mat x3D_C;
            if (cosParallaxRays < 0.9998)
            {
                // Linear Triangulation Method
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                cv::Mat w, u, vt;
                cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if (x3D.at<float>(3) == 0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
                // x3D = x3D / height * 1.69;
                // float distance = sqrt(pow(x3D.at<float>(0) - Ow1.at<float>(0, 0), 2) + pow(x3D.at<float>(1) - Ow1.at<float>(1, 0), 2) + pow(x3D.at<float>(2) - Ow1.at<float>(2, 0), 2));

                x3D_C = Rcw1 * x3D + tcw1;
                x3D_C = x3D_C / height * 1.69;
                float distance = sqrt(pow(x3D_C.at<float>(0), 2) + pow(x3D_C.at<float>(1), 2) + pow(x3D_C.at<float>(2), 2));
                // cout << x3D << " vs " << Ow1 << endl;
                dist_vec[k] = (distance);
                // dynamic_pos[k] = (x3D);
            }
            else
            {
                continue; //No stereo and very low parallax
            }
        }

        int cnt_avg = 0;
        float avg_dist = 0;
        for (int k = 0; k < dist_vec.size(); k++)
        {
            if (dist_vec[k] != -1)
            {
                avg_dist += dist_vec[k];
                cnt_avg++;
            }
        }
        avg_dist /= cnt_avg;

        if (cnt_avg)
        {
            BB_distance[i] = (avg_dist);
        }
        else
            continue;
    }

    cout << "BB_distance.size(): " << BB_distance.size() << endl;

    for (int i = 0; i < BB_distance.size(); i++)
    {
        cout << "ID: " << ID_1[i] << " dist : " << BB_distance[i] << endl;
    }

    cv::Mat src1 = mCurrentFrame.frameImg.clone();
    cv::cvtColor(src1, src1, cv::COLOR_GRAY2BGR);
    cv::Mat src2 = mPrevFrame.frameImg.clone();
    cv::cvtColor(src2, src2, cv::COLOR_GRAY2BGR);

    cv::Scalar olive(128, 128, 0), violet(221, 160, 221), brown(42, 42, 165);
    for (int i = 0; i < match_size; i++)
    {
        string ID_1_str = to_string(int(ID_1[i]));
        //cv::putText(src1, ID_1_str, Point2f(box_Pos_1[i].x, box_Pos_1[i].y + 60), 0, 2, violet, 2);

        string carDist = to_string(int(BB_distance[i]));
        cv::putText(src1, carDist, Point2f(box_Pos_1[i].x, box_Pos_1[i].y), 0, 2, cv::Scalar(0, 0, 255), 2);

        for (int k = 0; k < track_pt1_set.size(); k++)
        {
            circle(src1, track_pt1_set[i][k], 8, Scalar(255, 0, 0), -1);
        }
    }

    for (int i = 0; i < mCurrentFrame.boundingBoxes.size(); i++)
    {
        for (int j = 0; j < mPrevFrame.boundingBoxes.size(); j++)
        {
            float lenw1_BB = (mCurrentFrame.boundingBoxes[i].bbox_R - mCurrentFrame.boundingBoxes[i].bbox_L) / 2;
            float lenh1_BB = (mCurrentFrame.boundingBoxes[i].bbox_B - mCurrentFrame.boundingBoxes[i].bbox_T) / 2;
            float lenw2_BB = (mPrevFrame.boundingBoxes[j].bbox_R - mPrevFrame.boundingBoxes[j].bbox_L) / 2;
            float lenh2_BB = (mPrevFrame.boundingBoxes[j].bbox_B - mPrevFrame.boundingBoxes[j].bbox_T) / 2;

            // float diff_volume = abs((lenw1_BB * 2 * lenh1_BB * 2) - (lenw2_BB * 2 * lenh2_BB * 2));
            int k_idx = 0;
            if (mCurrentFrame.boundingBoxes[i].trackingID == mPrevFrame.boundingBoxes[j].trackingID)
            {
                cv::Rect rect1(cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_L, mCurrentFrame.boundingBoxes[i].bbox_T), cv::Point2f(mCurrentFrame.boundingBoxes[i].bbox_R, mCurrentFrame.boundingBoxes[i].bbox_B));
                cv::rectangle(src1, rect1, cv::Scalar(0, 0, 255), 2);
                cv::Rect rect2(cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_L, mPrevFrame.boundingBoxes[j].bbox_T), cv::Point2f(mPrevFrame.boundingBoxes[j].bbox_R, mPrevFrame.boundingBoxes[j].bbox_B));
                cv::rectangle(src2, rect2, cv::Scalar(0, 0, 255), 2);
            }
            else
                continue;
        }
    }
    cv::resize(src1, src1, cv::Size(src1.cols / 3, src1.rows / 3), 0, 0, CV_INTER_NN);
    imshow("cur", src1);
    // cv::resize(src2, src2, cv::Size(src2.cols / 3, src2.rows / 3), 0, 0, CV_INTER_NN);
    // imshow("prev", src2);
    cvWaitKey(1);
}

float Tracking::ComputeCameraHeight()
{
    cv::Mat groundPlane = mpMap->GetGroundPlane();

    if (!groundPlane.empty())
    {
        return Distance(groundPlane, mCurrentFrame.GetCameraCenter());
    }
}

float Tracking::Distance(cv::Mat plane, cv::Mat point)
{
    return fabs(plane.at<float>(0) * point.at<float>(0) +
                plane.at<float>(1) * point.at<float>(1) +
                plane.at<float>(2) * point.at<float>(2) + plane.at<float>(3));
}

} // namespace ORB_SLAM2
