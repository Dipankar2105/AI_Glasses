import pathlib

kconfig_path = pathlib.Path(r"C:\Users\Routewise\xiaozhi-esp32\main\Kconfig.projbuild")
cmakelists_path = pathlib.Path(r"C:\Users\Routewise\xiaozhi-esp32\main\CMakeLists.txt")

# Update Kconfig.projbuild
kc_text = kconfig_path.read_text(encoding="utf-8")
if "CONFIG_BOARD_TYPE_SEEED_XIAO_S3_SENSE" not in kc_text and "BOARD_TYPE_SEEED_XIAO_S3_SENSE" not in kc_text:
    target_kc = "    config BOARD_TYPE_SEEED_STUDIO_SENSECAP_WATCHER"
    insert_kc = """    config BOARD_TYPE_SEEED_XIAO_S3_SENSE
        bool "Seeed Studio XIAO ESP32-S3 Sense"
        depends on IDF_TARGET_ESP32S3
"""
    if target_kc in kc_text:
        kc_text = kc_text.replace(target_kc, insert_kc + target_kc, 1)
        kconfig_path.write_text(kc_text, encoding="utf-8")
        print("Updated Kconfig.projbuild successfully")
    else:
        print("Target not found in Kconfig.projbuild")
else:
    print("Already present in Kconfig.projbuild")

# Update CMakeLists.txt
cm_text = cmakelists_path.read_text(encoding="utf-8")
if "CONFIG_BOARD_TYPE_SEEED_XIAO_S3_SENSE" not in cm_text:
    target_cm = "elseif(CONFIG_BOARD_TYPE_SEEED_STUDIO_SENSECAP_WATCHER)"
    insert_cm = """elseif(CONFIG_BOARD_TYPE_SEEED_XIAO_S3_SENSE)
    set(BOARD_DIR "seeed-xiao-s3-sense")
"""
    if target_cm in cm_text:
        cm_text = cm_text.replace(target_cm, insert_cm + target_cm, 1)
        cmakelists_path.write_text(cm_text, encoding="utf-8")
        print("Updated CMakeLists.txt successfully")
    else:
        print("Target not found in CMakeLists.txt")
board_dir = pathlib.Path(r"C:\Users\Routewise\xiaozhi-esp32\main\boards\seeed-xiao-s3-sense")
board_dir.mkdir(parents=True, exist_ok=True)

config_json = """{
    "type": "seeed-xiao-s3-sense",
    "target": "esp32s3",
    "builds": [
        {
            "name": "seeed-xiao-s3-sense",
            "build_options": {
                "camera_hmirror": false,
                "camera_vflip": false
            }
        }
    ]
}
"""
(board_dir / "config.json").write_text(config_json, encoding="utf-8")

config_h = """#ifndef _BOARD_CONFIG_H_
#define _BOARD_CONFIG_H_

#include <driver/gpio.h>

#define AUDIO_INPUT_SAMPLE_RATE  16000
#define AUDIO_OUTPUT_SAMPLE_RATE 24000

// XIAO ESP32-S3 Sense onboard PDM Mic & external I2S DAC (MAX98357A)
#define AUDIO_I2S_MIC_GPIO_SCK  GPIO_NUM_42  // PDM CLK
#define AUDIO_I2S_MIC_GPIO_DIN  GPIO_NUM_41  // PDM DATA
#define AUDIO_I2S_SPK_GPIO_BCLK GPIO_NUM_1   // D0 (BCLK)
#define AUDIO_I2S_SPK_GPIO_LRCK GPIO_NUM_2   // D1 (LRCK / WS)
#define AUDIO_I2S_SPK_GPIO_DOUT GPIO_NUM_3   // D2 (DIN / DOUT)

#define BUILTIN_LED_GPIO        GPIO_NUM_21
#define BOOT_BUTTON_GPIO        GPIO_NUM_0
#define TOUCH_BUTTON_GPIO       GPIO_NUM_4   // D3 (Capacitive Touch)

// Seeed Studio XIAO ESP32-S3 Sense Camera (OV3660 / OV2640 DVP Mezzanine)
#define CAMERA_PIN_D0 GPIO_NUM_15
#define CAMERA_PIN_D1 GPIO_NUM_17
#define CAMERA_PIN_D2 GPIO_NUM_18
#define CAMERA_PIN_D3 GPIO_NUM_16
#define CAMERA_PIN_D4 GPIO_NUM_14
#define CAMERA_PIN_D5 GPIO_NUM_12
#define CAMERA_PIN_D6 GPIO_NUM_11
#define CAMERA_PIN_D7 GPIO_NUM_48
#define CAMERA_PIN_XCLK GPIO_NUM_10
#define CAMERA_PIN_PCLK GPIO_NUM_13
#define CAMERA_PIN_VSYNC GPIO_NUM_38
#define CAMERA_PIN_HREF GPIO_NUM_47
#define CAMERA_PIN_SIOC GPIO_NUM_40  // SCCB SCL
#define CAMERA_PIN_SIOD GPIO_NUM_39  // SCCB SDA
#define CAMERA_PIN_PWDN GPIO_NUM_NC
#define CAMERA_PIN_RESET GPIO_NUM_NC
#define XCLK_FREQ_HZ 20000000

#endif // _BOARD_CONFIG_H_
"""
(board_dir / "config.h").write_text(config_h, encoding="utf-8")

board_cc = """#include "wifi_board.h"
#include "codecs/no_audio_codec.h"
#include "system_reset.h"
#include "application.h"
#include "button.h"
#include "config.h"
#include "esp32_camera.h"
#include "mcp_server.h"
#include "led/single_led.h"

#include <esp_log.h>
#include <driver/gpio.h>

#define TAG "SeeedXiaoS3SenseBoard"

class SeeedXiaoS3SenseBoard : public WifiBoard {
private:
    Button boot_button_;
    Esp32Camera* camera_;

    void InitializeCamera() {
        camera_config_t config = {};
        config.pin_d0 = CAMERA_PIN_D0;
        config.pin_d1 = CAMERA_PIN_D1;
        config.pin_d2 = CAMERA_PIN_D2;
        config.pin_d3 = CAMERA_PIN_D3;
        config.pin_d4 = CAMERA_PIN_D4;
        config.pin_d5 = CAMERA_PIN_D5;
        config.pin_d6 = CAMERA_PIN_D6;
        config.pin_d7 = CAMERA_PIN_D7;
        config.pin_xclk = CAMERA_PIN_XCLK;
        config.pin_pclk = CAMERA_PIN_PCLK;
        config.pin_vsync = CAMERA_PIN_VSYNC;
        config.pin_href = CAMERA_PIN_HREF;
        config.pin_sccb_sda = CAMERA_PIN_SIOD;
        config.pin_sccb_scl = CAMERA_PIN_SIOC;
        config.sccb_i2c_port = 0;
        config.pin_pwdn = CAMERA_PIN_PWDN;
        config.pin_reset = CAMERA_PIN_RESET;
        config.xclk_freq_hz = XCLK_FREQ_HZ;
        config.pixel_format = PIXFORMAT_RGB565;
        config.frame_size = FRAMESIZE_VGA;
        config.jpeg_quality = 12;
        config.fb_count = 2;
        config.fb_location = CAMERA_FB_IN_PSRAM;
        config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
        camera_ = new Esp32Camera(config);
    }

    void InitializeButtons() {
        boot_button_.OnClick([this]() {
            auto& app = Application::GetInstance();
            if (app.GetDeviceState() == kDeviceStateStarting) {
                EnterWifiConfigMode();
                return;
            }
            app.ToggleChatState();
        });
    }

public:
    SeeedXiaoS3SenseBoard() :
        boot_button_(BOOT_BUTTON_GPIO),
        camera_(nullptr) {
        InitializeButtons();
        InitializeCamera();
    }

    virtual Led* GetLed() override {
        static SingleLed led(BUILTIN_LED_GPIO);
        return &led;
    }

    virtual AudioCodec* GetAudioCodec() override {
        static NoAudioCodecSimplexPdm audio_codec(AUDIO_INPUT_SAMPLE_RATE, AUDIO_OUTPUT_SAMPLE_RATE,
            AUDIO_I2S_SPK_GPIO_BCLK, AUDIO_I2S_SPK_GPIO_LRCK, AUDIO_I2S_SPK_GPIO_DOUT,
            AUDIO_I2S_MIC_GPIO_SCK, AUDIO_I2S_MIC_GPIO_DIN);
        return &audio_codec;
    }

    virtual Camera* GetCamera() override {
        return camera_;
    }
};

DECLARE_BOARD(SeeedXiaoS3SenseBoard);
"""
(board_dir / "seeed_xiao_s3_sense.cc").write_text(board_cc, encoding="utf-8")

readme = """# Seeed Studio XIAO ESP32-S3 Sense

## Features
- ESP32-S3FN8 / ESP32-S3R8 SoC (240MHz dual-core, 8MB Octal PSRAM)
- OV3660 / OV2640 Camera via Mezzanine Connector
- Onboard PDM Microphone (CLK GPIO 42, DATA GPIO 41)
- External MAX98357A I2S DAC / Speaker (BCLK GPIO 1, LRCK GPIO 2, DIN GPIO 3)
- Headless / Smart Glasses operation with Vision MCP Tool (`self.camera.take_photo`)

## Pin Mapping
- Camera D0..D7: GPIO 15, 17, 18, 16, 14, 12, 11, 48
- Camera XCLK: GPIO 10, PCLK: GPIO 13, VSYNC: GPIO 38, HREF: GPIO 47
- Camera SCCB I2C: SIOC/SCL GPIO 40, SIOD/SDA GPIO 39
- Mic PDM: CLK GPIO 42, DATA GPIO 41
- Speaker I2S: BCLK GPIO 1, LRCK GPIO 2, DOUT GPIO 3
"""
(board_dir / "README.md").write_text(readme, encoding="utf-8")
print("Board files written without BOM")

