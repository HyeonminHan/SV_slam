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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include <fstream>

#include <mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular) : mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
                                                                mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

LocalMapping::LocalMapping(Map *pMap, const float bMonocular, cv::Mat roadMask) : roadMask(roadMask), mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
                                                                mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker = pTracker;
}

void LocalMapping::Run()
{
    mbFinished = false;

    while (1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if (CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            // Check recent MapPoints
            MapPointCulling();

            // Triangulate new MapPoints
            // cout << "before CreateNewMapPoints: " << mpMap->GetAllMapPoints().size() << endl;
            CreateNewMapPoints();
            // cout << "after CreateNewMapPoints: " << mpMap->GetAllMapPoints().size() << endl;

            if (!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

            mbAbortBA = false;

            // cout << "mpMap->KeyFramesInMap():: " << mpMap->KeyFramesInMap()<< endl;
            if (!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                if (mpMap->KeyFramesInMap() > 2)
                {
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);
                }
                // Check redundant local Keyframes
                KeyFrameCulling();
            }

            /* make Plane MP */
            SetGroundPoint();

            mpMap->SetGroundPoint(mpCurrentKeyFrame->groundPoints);

            int nGPP = mpCurrentKeyFrame->groundPoints.size();

            if (nGPP > 20)
            {
                if (ComputeGroundPlane(nGPP))
                {
                    // cout << "Compute ground plane." << endl;
                    mpMap->SetGroundPlane(mpCurrentKeyFrame->groundPlane);
                }
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if (Stop())
        {
            // Safe area to stop
            while (isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if (CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if (CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;
}

bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return (!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for (size_t i = 0; i < vpMapPointMatches.size(); i++)
    {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP)
        {
            if (!pMP->isBad())
            {
                if (!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if (mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while (lit != mlpRecentAddedMapPoints.end())
    {
        MapPoint *pMP = *lit;
        if (pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (pMP->GetFoundRatio() < 0.25f)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

/* create GP */
void LocalMapping::SetGroundPoint()
{
    //for문 돌며 전체 MP 가져오기 후보 KF에서
    //가져온 MP에서 영상으로 투영됬을때 올바를 것들에 해당하여 RIO영역 컬링을 수행
    //수행된 MP를 대상으로 addGP를 keyF에 수행
    //수행 결과는 keyF의 vec<MP>

    for (int i = 0; i < mpCurrentKeyFrame->N; i++)
    {
        MapPoint *pMP = mpCurrentKeyFrame->GetMapPoint(i);
        if (pMP)
        {
            if (pMP->Observations() > 0)
            {
                // if (IsInRoI(1920, 1080, mpCurrentKeyFrame->mvKeys[i]))
                //     mpCurrentKeyFrame->AddGroundPoint(pMP);
                if (IsInRoI(mpCurrentKeyFrame->mvKeys[i]))
                    mpCurrentKeyFrame->AddGroundPoint(pMP);
            }
        }
    }
}

bool LocalMapping::ComputeGroundPlane(int nGPP)
{
    int ransacIt = 10;
    int maxCnt = 0;

    cv::Mat groundPlane = Compute3DPlaneRansac(nGPP, ransacIt, maxCnt);

    if (cv::norm(groundPlane) == 0)
    {
        return false;
    }

    NormalDirectionCorrection(groundPlane);

    if (!prevGroundPlane.empty())
    {
        float angle = Angle(groundPlane, prevGroundPlane);
        // cout << "angle : " << angle << endl;

        if (angle > 20)
        {
            mpCurrentKeyFrame->groundPlane = prevGroundPlane;
            // cout << "Ground plane outlier" << endl;
        }
        else
        {
            mpCurrentKeyFrame->groundPlane = groundPlane;
            prevGroundPlane = groundPlane;
        }
    }
    else
    {
        mpCurrentKeyFrame->groundPlane = groundPlane;
        prevGroundPlane = groundPlane;
    }

    return true;
}

float LocalMapping::Angle(cv::Mat plane1, cv::Mat plane2)
{
    cv::Mat normal1 = plane1.rowRange(0, 3);
    cv::Mat normal2 = plane2.rowRange(0, 3);

    return acos(normal1.dot(normal2)) * 180.0 / M_PI;
}

cv::Mat LocalMapping::Compute3DPlaneRansac(int nGPP, int ransacIt, int &maxCnt_)
{
    int maxCnt = 0;
    cv::Mat maxGP;
    for (int i = 0; i < ransacIt; i++)
    {
        vector<int> randNums;

        for (int i = 0; i < 3; i++)
        {
            int num = rand() % nGPP;
            if (find(randNums.begin(), randNums.end(), num) != randNums.end())
            {
                i--;
                continue;
            }
            randNums.push_back(num);
        }

        cv::Mat tempGP = Compute3DPlane(randNums);
        int cnt = 0;
        for (int j = 0; j < nGPP; j++)
        {
            float error = Distance(tempGP, mpCurrentKeyFrame->groundPoints[j]->GetWorldPos());
            if (error < 0.01)
            {
                cnt++;
            }
        }
        if (cnt > nGPP * 0.75)
        {
            maxCnt = cnt;
            maxGP = tempGP;
            break;
        }
        if (cnt > maxCnt)
        {
            maxCnt = cnt;
            maxGP = tempGP;
        }
    }

    maxCnt_ = maxCnt;
    return maxGP;
}

cv::Mat LocalMapping::Compute3DPlane(vector<int> randNums)
{
    std::vector<MapPoint *> mps = mpCurrentKeyFrame->groundPoints;
    cv::Mat p0 = mps[randNums[0]]->GetWorldPos();
    cv::Mat p1 = mps[randNums[1]]->GetWorldPos();
    cv::Mat p2 = mps[randNums[2]]->GetWorldPos();
    cv::Mat v1 = p1 - p0;
    cv::Mat v2 = p2 - p0;
    cv::Mat normal;
    normalize(v1.cross(v2), normal);

    cv::Mat plane(4, 1, CV_32F);
    plane.at<float>(0) = normal.at<float>(0);
    plane.at<float>(1) = normal.at<float>(1);
    plane.at<float>(2) = normal.at<float>(2);
    plane.at<float>(3) = -1 * p0.dot(normal);

    return plane;
}

float LocalMapping::Distance(cv::Mat plane, cv::Mat point)
{
    return fabs(plane.at<float>(0) * point.at<float>(0) +
                plane.at<float>(1) * point.at<float>(1) +
                plane.at<float>(2) * point.at<float>(2) + plane.at<float>(3));
}

void LocalMapping::NormalDirectionCorrection(cv::Mat &plane)
{
    cv::Mat normal = plane.rowRange(0, 3);
    cv::Mat up = (cv::Mat_<float>(3, 1) << 0.f, -1.f, 0.f);
    float cosTheta = normal.dot(up);
    if (cosTheta < 0)
    {
        plane = -1 * plane;
    }
}

// bool LocalMapping::IsInRoI(int width_, int height_, cv::KeyPoint point)
// {
//     int width = width_;
//     int height = height_;

//     // cv::Point pt_1(width/3, height/4*3);
//     // cv::Point pt_2(width/3*2, height/4*3);
//     // cv::Point pt_3(width/3, height/3*3);
//     // cv::Point pt_4(width/3*2, height/3*3);

//     if (point.pt.x > width / 3 && point.pt.x < width / 3 * 2 && point.pt.y > height / 4 * 3 && point.pt.y < height / 3 * 2 * 3)
//         return true;
//     else
//         return false;
// }

bool LocalMapping::IsInRoI(cv::KeyPoint point)
{
    if (roadMask.at<uchar>(point.pt.y, point.pt.x) == 0)
        return true;
    else
        return false;
}

float LocalMapping::get_height(cv::Mat cam_center, std::vector<MapPoint *> GP_vec)
{
    std::vector<MapPoint *> temp = GP_vec;

    float h_ = 0;
    int cnt = 0;
    for (size_t i = 0; i < temp.size(); i++)
    {
        cv::Mat pos = temp[i]->GetWorldPos();
        h_ += pos.at<float>(1);
        cnt++;
    }

    h_ = -(h_ / cnt);

    float cam_c = -(cam_center.at<float>(1));
    float h_get = abs(cam_c - h_);

    // cout << "cam: " <<cam_c  << " , h_: " << h_ << " , h_get: " << h_get << endl;

    return h_get;
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if (mbMonocular)
        nn = 20;
    const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn); //K2

    ORBmatcher matcher(0.6, false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0, 3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

    int nnew = 0;

    // Search matches with epipolar restriction and triangulate
    for (size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        if (i > 0 && CheckNewKeyFrames())
            return;

        KeyFrame *pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2 - Ow1;
        const float baseline = cv::norm(vBaseline);

        if (!mbMonocular)
        {
            if (baseline < pKF2->mb)
                continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline / medianDepthKF2;

            if (ratioBaselineDepth < 0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t, size_t>> vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0, 3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for (int ikp = 0; ikp < nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur >= 0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur >= 0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

            cv::Mat ray1 = Rwc1 * xn1;
            cv::Mat ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));
            // cout << "cosParallaxRays:: " << cosParallaxRays << endl;

            cv::Mat x3D;
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
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
            if (z1 <= 0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
            if (z2 <= 0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
            const float invz1 = 1.0 / z1;

            if (!bStereo1)
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                {
                    // cout << "repjoj err1 " << errX1*errX1+errY1*errY1<< endl;
                    continue;
                }
            }
            else
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
            const float invz2 = 1.0 / z2;
            if (!bStereo2)
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                {
                    cout << "repjoj err2 " << errX2 * errX2 + errY2 * errY2 << endl;
                    continue;
                }
            }
            else
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf * invz2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D - Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D - Ow2;
            float dist2 = cv::norm(normal2);

            if (dist1 == 0 || dist2 == 0)
                continue;

            const float ratioDist = dist2 / dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                continue;

            /* track on off */
            // Triangulation is succesfull
            MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

            pMP->AddObservation(mpCurrentKeyFrame, idx1);
            pMP->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
            pKF2->AddMapPoint(pMP, idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if (mbMonocular)
        nn = 20;
    const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame *> vpTargetKFs;
    for (vector<KeyFrame *>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
    {
        KeyFrame *pKFi = *vit;
        if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame *> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for (vector<KeyFrame *>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
        {
            KeyFrame *pKFi2 = *vit2;
            if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId || pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (vector<KeyFrame *>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
    {
        KeyFrame *pKFi = *vit;

        matcher.Fuse(pKFi, vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint *> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

    for (vector<KeyFrame *>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++)
    {
        KeyFrame *pKFi = *vitKF;

        vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for (vector<MapPoint *>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP; vitMP++)
        {
            MapPoint *pMP = *vitMP;
            if (!pMP)
                continue;
            if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
    {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP)
        {
            if (!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w * R2w.t();
    cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    return K1.t().inv() * t12x * R12 * K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if (mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if (mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames = flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if (flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame *> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for (vector<KeyFrame *>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
    {
        KeyFrame *pKF = *vit;
        if (pKF->mnId == 0)
            continue;
        const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs = nObs;
        int nRedundantObservations = 0;
        int nMPs = 0;
        for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
        {
            MapPoint *pMP = vpMapPoints[i];
            if (pMP)
            {
                if (!pMP->isBad())
                {
                    if (!mbMonocular)
                    {
                        if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
                            continue;
                    }

                    nMPs++;
                    if (pMP->Observations() > thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                        int nObs = 0;
                        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
                        {
                            KeyFrame *pKFi = mit->first;
                            if (pKFi == pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if (scaleLeveli <= scaleLevel + 1)
                            {
                                nObs++;
                                if (nObs >= thObs)
                                    break;
                            }
                        }
                        if (nObs >= thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if (nRedundantObservations > 0.9 * nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2), 0, -v.at<float>(0),
            -v.at<float>(1), v.at<float>(0), 0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while (1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if (mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested = false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} // namespace ORB_SLAM2
