# Pin Mapping Specification — Seeed Studio XIAO ESP32-S3 Sense

> **IMPORTANT**: This pin map documents the hardware pin assignments for the Seeed Studio XIAO ESP32-S3 Sense board, including factory-integrated camera/mic mezzanine and external breakout pins.

## 1. Integrated Board Connections (Camera & Sense Expansion)

The XIAO ESP32-S3 Sense daughterboard integrates the OV3660 camera and digital I2S microphone directly via a high-density B2B connector:

| Subsystem | Signal Name | ESP32-S3 Internal GPIO | Notes |
| :--- | :--- | :--- | :--- |
| **OV3660 Camera** | `CAM_D0` | GPIO 15 | 8-bit parallel DVP bus |
| | `CAM_D1` | GPIO 17 | |
| | `CAM_D2` | GPIO 18 | |
| | `CAM_D3` | GPIO 16 | |
| | `CAM_D4` | GPIO 14 | |
| | `CAM_D5` | GPIO 12 | |
| | `CAM_D6` | GPIO 11 | |
| | `CAM_D7` | GPIO 48 | |
| | `CAM_XCLK` | GPIO 10 | Master clock 20MHz |
| | `CAM_PCLK` | GPIO 13 | Pixel clock |
| | `CAM_VSYNC`| GPIO 38 | Vertical frame sync |
| | `CAM_HREF` | GPIO 47 | Horizontal reference |
| | `CAM_SIOC` | GPIO 40 | SCCB I2C Clock |
| | `CAM_SIOD` | GPIO 39 | SCCB I2C Data |
| | `CAM_RESET`| GPIO -1 | Tied to system reset / pull-up |
| | `CAM_PWDN` | GPIO -1 | Tied to GND (always powered) |
| **I2S Mic (Sense)**| `MIC_CLK` | GPIO 42 | I2S Bit Clock / PDM CLK |
| | `MIC_DATA` | GPIO 41 | I2S PDM Data Input |
| **MicroSD (Sense)**| `SD_CS` | GPIO 21 | SPI / SDIO CS |
| | `SD_MOSI` | GPIO 9 | SPI MOSI / SDIO CMD |
| | `SD_MISO` | GPIO 8 | SPI MISO / SDIO D0 |
| | `SD_SCK` | GPIO 7 | SPI SCK / SDIO CLK |

---

## 2. External Header Pin Assignments (Breakout Headers)

External peripheral breakouts for I2S DAC/Speaker, MPU-6050 IMU, and Capacitive Touch:

| Pin Label | ESP32-S3 GPIO | Assigned Peripheral Function | Notes |
| :--- | :--- | :--- | :--- |
| **D0** | GPIO 1 | `I2S_SPK_BCLK` | MAX98357A I2S Bit Clock |
| **D1** | GPIO 2 | `I2S_SPK_LRCK` (WS) | MAX98357A I2S Word Select |
| **D2** | GPIO 3 | `I2S_SPK_DIN` (DATA) | MAX98357A I2S Serial Data Out |
| **D3** | GPIO 4 | `TOUCH_PAD` | Capacitive Touch Sensor (Touch4 ADC) |
| **D4** | GPIO 5 | `I2C_SDA` | MPU-6050 IMU I2C Data (400kHz pull-up) |
| **D5** | GPIO 6 | `I2C_SCL` | MPU-6050 IMU I2C Clock |
| **3V3** | 3.3V Rail | Regulated 3.3V Power Out | Max 500mA LDO output |
| **GND** | GND | Common System Ground | |
| **BAT+** | VBAT | 1S LiPo Positive Terminal | Charge management onboard |
