//
// Created by mlompart on 25.06.22.
//

#pragma once
#include <cinttypes>
#include <string>
namespace camera
{

    class CameraStream
    {
    public:
        CameraStream(uint32_t captureWidth, uint32_t captureHeight,
                     uint32_t displayWidth, uint32_t displayHeight,
                     uint32_t frameRate, uint32_t flipMethod);
        CameraStream();
        std::string init() const;
        uint32_t getCaptureWidth() const;
        uint32_t getCaptureHeight() const;
        uint32_t getDisplayWidth() const;
        uint32_t getDisplayHeight() const;
        uint32_t getFrameRate() const;
        uint32_t getFlipMethod() const;

        void setCaptureWidth(uint32_t captureWidth);
        void setCaptureHeight(uint32_t captureHeight);
        void setDisplayWidth(uint32_t displayWidth);
        void setDisplayHeight(uint32_t displayHeight_);
        void setFrameRate(uint32_t frameRate);
        void setFlipMethod(uint32_t flipMethod);

    private:
        uint32_t captureWidth_{};
        uint32_t captureHeight_{};
        uint32_t displayWidth_{};
        uint32_t displayHeight_{};
        uint32_t frameRate_{};
        uint32_t flipMethod_{};
    };

}// namespace camera
