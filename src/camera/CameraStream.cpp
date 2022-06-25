//
// Created by mlompart on 25.06.22.
//

#include "CameraStream.hpp"
namespace camera
{
    CameraStream::CameraStream(uint32_t captureWidth, uint32_t captureHeight,
                               uint32_t displayWidth, uint32_t displayHeight,
                               uint32_t frameRate, uint32_t flipMethod) : captureWidth_(captureWidth),
                                                                          captureHeight_(captureHeight),
                                                                          displayWidth_(displayWidth),
                                                                          displayHeight_(displayHeight),
                                                                          frameRate_(frameRate),
                                                                          flipMethod_(flipMethod){};
    CameraStream::CameraStream(): captureWidth_(1280),
                                   captureHeight_(720),
                                   displayWidth_(1280),
                                   displayHeight_(720),
                                   frameRate_(30),
                                   flipMethod_(0){};
    std::string CameraStream::init() const
    {
        return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(captureWidth_) + ", height=(int)" +
               std::to_string(captureHeight_) + ", framerate=(fraction)" + std::to_string(frameRate_) +
               "/1 ! nvvidconv flip-method=" + std::to_string(flipMethod_) + " ! video/x-raw, width=(int)" +
               std::to_string(displayWidth_) + ", height=(int)" + std::to_string(displayHeight_) +
               ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
    }
    uint32_t CameraStream::getCaptureWidth() const { return captureWidth_; }
    uint32_t CameraStream::getCaptureHeight() const { return captureHeight_; }
    uint32_t CameraStream::getDisplayWidth() const { return displayWidth_; }
    uint32_t CameraStream::getDisplayHeight() const { return displayHeight_; }
    uint32_t CameraStream::getFrameRate() const { return frameRate_; }
    uint32_t CameraStream::getFlipMethod() const { return flipMethod_; }

    void CameraStream::setCaptureWidth(uint32_t captureWidth) { captureWidth_ = captureWidth; }
    void CameraStream::setCaptureHeight(uint32_t captureHeight) { captureHeight_ = captureHeight; }
    void CameraStream::setDisplayWidth(uint32_t displayWidth) { displayWidth_ = displayWidth; }
    void CameraStream::setDisplayHeight(uint32_t displayHeight) { displayHeight_ = displayHeight; }
    void CameraStream::setFrameRate(uint32_t frameRate) { frameRate_ = frameRate; }
    void CameraStream::setFlipMethod(uint32_t flipMethod) { flipMethod_ = flipMethod; }
}// namespace camera