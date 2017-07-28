/*
 * dressup - openpose example
 * shows clothes on some skeleton parts
 *
 * based on https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/examples/tutorial_pose/2_extract_pose_or_heatmat_from_image.cpp
 *
 *
 */
#include <string>
// 3rdparty dependencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h>  // google::InitGoogleLogging
// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <opencv2/opencv.hpp>

// Debugging
DEFINE_int32(logging_level, 3, "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                               " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                               " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_path, "examples/media/COCO_val2014_000000000192.jpg", "Process the desired image.");
// OpenPose
DEFINE_string(model_pose, "COCO", "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_string(model_folder, "models/", "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution, "656x368", "Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased,"
                                         " the speed increases. For maximum speed-accuracy balance, it should keep the closest aspect"
                                         " ratio possible to the images or videos to be processed. E.g. the default `656x368` is"
                                         " optimal for 16:9 videos, e.g. full HD (1980x1080) and HD (1280x720) videos.");
DEFINE_string(resolution, "1280x720", "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                      " default images resolution.");
DEFINE_int32(num_gpu_start, 0, "GPU device start number.");
DEFINE_double(scale_gap, 0.3, "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                              " If you want to change the initial scale, you actually want to multiply the"
                              " `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number, 1, "Number of scales to average.");
// OpenPose Rendering
DEFINE_int32(part_to_show, 19, "Part to show from the start.");
DEFINE_bool(disable_blending, false, "If blending is enabled, it will merge the results with the original frame. If disabled, it"
                                     " will only display the results.");
DEFINE_double(render_threshold, 0.05, "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                      " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                      " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                      " more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                               " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap, 0.7, "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
                                  " heatmap, 0 will only show the frame. Only valid for GPU rendering.");

cv::VideoCapture cap; // webcam capture

int Neck = 1;      // index of neck
int RShoulder = 2; // index of right shoulder
int LShoulder = 5; // index of left shoulder
int RHip = 8;      // index of right hip
int LHip = 11;     // index of left hip

cv::Mat scarf;  // picture of a scarf
cv::Mat skirt;  // picture of a skirt
cv::Mat tshirt; // picture of a tshirt

/*
 * source: http://jepsonsblog.blogspot.de/2012/10/overlay-transparent-image-in-opencv.html
 */
void overlayImage(const cv::Mat &background, const cv::Mat &foreground,
                  cv::Mat &output, cv::Point2i location)
{
    background.copyTo(output);

    // start at the row indicated by location, or at row 0 if location.y is negative.
    for (int y = std::max(location.y, 0); y < background.rows; ++y)
    {
        int fY = y - location.y; // because of the translation

        // we are done of we have processed all rows of the foreground image.
        if (fY >= foreground.rows)
            break;

        // start at the column indicated by location,

        // or at column 0 if location.x is negative.
        for (int x = std::max(location.x, 0); x < background.cols; ++x)
        {
            int fX = x - location.x; // because of the translation.

            // we are done with this row if the column is outside of the foreground image.
            if (fX >= foreground.cols)
                break;

            // determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
            double opacity =
                ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

                / 255.;

            // and now combine the background and foreground pixel, using the opacity,

            // but only if opacity > 0.
            for (int c = 0; opacity > 0 && c < output.channels(); ++c)
            {
                unsigned char foregroundPx =
                    foreground.data[fY * foreground.step + fX * foreground.channels() + c];
                unsigned char backgroundPx =
                    background.data[y * background.step + x * background.channels() + c];
                output.data[y * output.step + output.channels() * x + c] =
                    backgroundPx * (1. - opacity) + foregroundPx * opacity;
            }
        }
    }
}

/**
 * @brief renderScarf
 * renders scarf to the frame
 * @param frame
 * @param topleft
 * position of topleft scarf corner on frame
 * @param topright
 * position of topright scarf corner on frame
 */
void renderScarf(cv::Mat frame, cv::Point2f topleft, cv::Point2f topright)
{
    if (scarf.empty())
        return;
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];

    srcTri[0] = cv::Point2f(0, 0);
    srcTri[1] = cv::Point2f(scarf.cols - 1, 0);
    srcTri[2] = cv::Point2f(0, scarf.rows - 1);

    dstTri[0] = topleft;
    dstTri[1] = topright;
    dstTri[2] = cv::Point2f(topleft.x, topleft.y + (topright.x - topleft.x));

    cv::Mat warp_mat(2, 3, CV_32FC1);
    warp_mat = cv::getAffineTransform(srcTri, dstTri);

    cv::Mat warpDest = cv::Mat(frame.cols, frame.rows, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    cv::warpAffine(scarf, warpDest, warp_mat, frame.size());
    overlayImage(frame, warpDest, frame, cv::Point2i(0, 0));
}

/**
 * @brief renderSkirt
 * renders skirt to the frame
 * @param frame
 * @param topleft
 * @param topright
 */
void renderSkirt(cv::Mat frame, cv::Point2f topleft, cv::Point2f topright)
{
    if (scarf.empty())
        return;
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];

    srcTri[0] = cv::Point2f(0, 0);
    srcTri[1] = cv::Point2f(skirt.cols - 1, 0);
    srcTri[2] = cv::Point2f(0, skirt.rows - 1);

    dstTri[0] = topleft;
    dstTri[1] = topright;
    dstTri[2] = cv::Point2f(topleft.x, topleft.y + (topright.x - topleft.x));

    cv::Mat warp_mat(2, 3, CV_32FC1);
    warp_mat = cv::getAffineTransform(srcTri, dstTri);

    cv::Mat warpDest = cv::Mat(frame.cols, frame.rows, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    cv::warpAffine(skirt, warpDest, warp_mat, frame.size());
    overlayImage(frame, warpDest, frame, cv::Point2i(0, 0));
}

/**
 * @brief renderTShirt
 * renders tshirt to the frame
 * @param frame
 * @param topleft
 * @param topright
 * @param bottomleft
 */
void renderTShirt(cv::Mat frame, cv::Point2f topleft, cv::Point2f topright, cv::Point2f bottomleft)
{
    if (scarf.empty())
        return;
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];

    srcTri[0] = cv::Point2f(0, 0);
    srcTri[1] = cv::Point2f(tshirt.cols - 1, 0);
    srcTri[2] = cv::Point2f(0, tshirt.rows - 1);

    dstTri[0] = topleft;
    dstTri[1] = topright;
    dstTri[2] = bottomleft;

    cv::Mat warp_mat(2, 3, CV_32FC1);
    warp_mat = cv::getAffineTransform(srcTri, dstTri);

    cv::Mat warpDest = cv::Mat(frame.cols, frame.rows, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    cv::warpAffine(tshirt, warpDest, warp_mat, frame.size());
    overlayImage(frame, warpDest, frame, cv::Point2i(0, 0));
}

/**
 * @brief renderKeypoints
 * renders keypoints to the frame and also clothes if enough keypoints are found
 * @param frame
 * the frame to render on
 * @param keypoints
 * found keypoints to render
 * @param pairs
 * @param colors
 * @param thicknessCircleRatio
 * @param thicknessLineRatioWRTCircle
 * @param threshold
 * @param scaleNetToFrame
 */
void renderKeypoints(cv::Mat frame, const op::Array<float> &keypoints, const std::vector<unsigned int> &pairs,
                     const std::vector<float> colors, const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                     const float threshold, float scaleNetToFrame)
{
    try
    {
        if (!frame.empty())
        {

            // Get frame channels
            const auto width = frame.size[2];
            const auto height = frame.size[1];
            const auto area = width * height;
            cv::Mat frameB{height, width, CV_32FC1, &frame.data[0]};
            cv::Mat frameG{height, width, CV_32FC1, &frame.data[area * sizeof(float) / sizeof(uchar)]};
            cv::Mat frameR{height, width, CV_32FC1, &frame.data[2 * area * sizeof(float) / sizeof(uchar)]};

            // Parameters
            const auto lineType = 8;
            const auto shift = 0;
            const auto numberColors = colors.size();
            const auto thresholdRectangle = 0.1f;
            const auto numberKeypoints = keypoints.getSize(1);

            // Keypoints
            for (auto person = 0; person < keypoints.getSize(0); person++)
            {
                const auto personRectangle = getKeypointsRectangle(keypoints, person, numberKeypoints, thresholdRectangle);
                if (personRectangle.area() > 0)
                {
                    const auto ratioAreas = op::fastMin(1.f, op::fastMax(personRectangle.width / (float)width, personRectangle.height / (float)height));
                    // Size-dependent variables
                    const auto thicknessRatio = op::fastMax(op::intRound(std::sqrt(area) * thicknessCircleRatio * ratioAreas), 2);
                    // Negative thickness in cv::circle means that a filled circle is to be drawn.
                    const auto thicknessCircle = (ratioAreas > 0.05 ? thicknessRatio : -1);
                    const auto thicknessLine = op::intRound(thicknessRatio * thicknessLineRatioWRTCircle);
                    const auto radius = thicknessRatio / 2;

                    cv::Point2f lshoulder(-1, -1);
                    cv::Point2f rshoulder(-1, -1);
                    cv::Point2f lhip(-1, -1);
                    cv::Point2f rhip(-1, -1);

                    // Draw circles
                    for (auto part = 0; part < numberKeypoints; part++)
                    {
                        const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                        if (keypoints[faceIndex + 2] > threshold)
                        {
                            const auto colorIndex = part * 3;
                            const cv::Scalar color{colors[colorIndex % numberColors],
                                                   colors[(colorIndex + 1) % numberColors],
                                                   colors[(colorIndex + 2) % numberColors]};
                            const cv::Point center{op::intRound(keypoints[faceIndex]), op::intRound(keypoints[faceIndex + 1])};

                            if (part == Neck || part == RShoulder || part == LShoulder || part == RHip || part == LHip)
                                cv::circle(frame, center /*scaleNetToFrame*/, 5, color, 3, lineType, shift);

                            if (part == LShoulder)
                            {
                                lshoulder = center;
                            }
                            if (part == RShoulder)
                            {
                                rshoulder = center;
                            }
                            if (part == LHip)
                            {
                                lhip = center;
                                lhip.x;
                            }
                            if (part == RHip)
                            {
                                rhip = center;
                                rhip.x;
                            }
                        }
                    }
                    if (rshoulder.x >= 0 && rshoulder.y >= 0 && lshoulder.x >= 0 && lshoulder.y >= 0 && rhip.x >= 0 && rhip.y >= 0)
                    {
                        renderTShirt(frame, rshoulder, lshoulder, rhip);
                    }
                    if (rshoulder.x >= 0 && rshoulder.y >= 0 && lshoulder.x >= 0 && lshoulder.y >= 0)
                    {
                        renderScarf(frame, rshoulder, lshoulder);
                    }
                    if (rhip.x >= 0 && rhip.y >= 0 && lhip.x >= 0 && lhip.y >= 0)
                    {
                        renderSkirt(frame, rhip, lhip);
                    }
                }
            }
        }
    }
    catch (const std::exception &e)
    {
    }
}

/**
 * @brief loadImages
 * loads semitransparent images of clothes.
 */
void loadImages()
{
    scarf = cv::imread("scarf.png", cv::IMREAD_UNCHANGED);
    skirt = cv::imread("skirt.png", cv::IMREAD_UNCHANGED);
    tshirt = cv::imread("tshirt.png", cv::IMREAD_UNCHANGED);
}

int dressup()
{
    if (!cap.open(0))
        return 1;
    loadImages();

    op::log("dressup - openpose example", op::Priority::High);
    // ------------------------- INITIALIZATION -------------------------
    // Step 1 - Set logging level
    // - 0 will output all the logging messages
    // - 255 will output nothing
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // Step 2 - Read Google flags (user defined configuration)
    // outputSize
    const auto outputSize = op::flagsToPoint(FLAGS_resolution, "1280x720");
    // netInputSize
    const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "656x368");
    // netOutputSize
    const auto netOutputSize = netInputSize;
    // poseModel
    const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
    // Check no contradictory flags enabled
    if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
        op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
    if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
        op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.", __LINE__, __FUNCTION__, __FILE__);
    // Logging
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // Step 3 - Initialize all required classes
    op::CvMatToOpInput cvMatToOpInput{netInputSize, FLAGS_scale_number, (float)FLAGS_scale_gap};
    op::CvMatToOpOutput cvMatToOpOutput{outputSize};
    op::PoseExtractorCaffe poseExtractorCaffe{netInputSize, netOutputSize, outputSize, FLAGS_scale_number, poseModel,
                                              FLAGS_model_folder, FLAGS_num_gpu_start};
    op::PoseRenderer poseRenderer{netOutputSize, outputSize, poseModel, nullptr, (float)FLAGS_render_threshold,
                                  !FLAGS_disable_blending, (float)FLAGS_alpha_pose};
    op::OpOutputToCvMat opOutputToCvMat{outputSize};
    const op::Point<int> windowedSize = outputSize;
    op::FrameDisplayer frameDisplayer{windowedSize, "OpenPose Tutorial - Example 1"};
    // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    poseExtractorCaffe.initializationOnThread();
    poseRenderer.initializationOnThread();

    // ------------------------- POSE ESTIMATION AND RENDERING -------------------------
    // Step 1 - Read and load image, error if empty (possibly wrong path)

    cv::Mat inputImage; // = op::loadImage(FLAGS_image_path, CV_LOAD_IMAGE_COLOR); // Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
    cap >> inputImage;
    if (inputImage.empty())
        op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
    // Step 2 - Format input image to OpenPose input and output formats
    op::Array<float> netInputArray;
    std::vector<float> scaleRatios;

    double scaleInputToOutput;
    op::Array<float> outputArray;

    const auto thicknessCircleRatio = 1.f / 75.f;
    const auto thicknessLineRatioWRTCircle = 0.75f;
    const auto &pairs = op::POSE_BODY_PART_PAIRS_RENDER[(int)poseModel];

    char c = 'q';

    do
    {
        cap >> inputImage;
        if (inputImage.empty())
            op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
        std::tie(netInputArray, scaleRatios) = cvMatToOpInput.format(inputImage);
        std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput.format(inputImage);

        // Step 3 - Estimate poseKeypoints
        poseExtractorCaffe.forwardPass(netInputArray, {inputImage.cols, inputImage.rows}, scaleRatios);
        op::Array<float> poseKeypoints = poseExtractorCaffe.getPoseKeypoints();
        const auto scaleNetToOutput = poseExtractorCaffe.getScaleNetToOutput();

        // Step 5 - OpenPose output format to cv::Mat
        auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);

        // ------------------------- SHOWING RESULT AND CLOTHES -------------------------

        renderKeypoints(outputImage, poseKeypoints, pairs, op::POSE_COLORS[(int)poseModel], thicknessCircleRatio,
                        thicknessLineRatioWRTCircle, (float)FLAGS_render_threshold, scaleNetToOutput);
        cv::imshow("dressup - openpose example", outputImage);
        c = cv::waitKey(100);
    } while (c != 'y' && c != 'n');

    op::log("Example successfully finished.", op::Priority::High);
    // Return successful message
    return 0;
}

int main(int argc, char *argv[])
{
    // Initializing google logging (Caffe uses it for logging)
    google::InitGoogleLogging("dressup - openpose example");

    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    return dressup();
}
