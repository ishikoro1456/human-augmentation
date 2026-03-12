#include <Wire.h>

namespace {

constexpr uint8_t kBnoAddrA = 0x28;
constexpr uint8_t kBnoAddrB = 0x29;

constexpr uint8_t kRegChipId = 0x00;
constexpr uint8_t kRegPageId = 0x07;
constexpr uint8_t kRegAccXLSB = 0x08;
constexpr uint8_t kRegUnitSel = 0x3B;
constexpr uint8_t kRegOprMode = 0x3D;
constexpr uint8_t kRegPwrMode = 0x3E;
constexpr uint8_t kRegSysTrigger = 0x3F;

constexpr uint8_t kChipId = 0xA0;

constexpr uint8_t kModeConfig = 0x00;
constexpr uint8_t kModeAmg = 0x07;
constexpr uint8_t kPowerNormal = 0x00;

constexpr uint32_t kBaudRate = 115200;
constexpr uint32_t kSampleIntervalMs = 20;

uint8_t g_bno_addr = 0;
bool g_bno_ready = false;

bool write8(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(g_bno_addr);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readLen(uint8_t reg, uint8_t* out, size_t len) {
  Wire.beginTransmission(g_bno_addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }
  size_t read_len = Wire.requestFrom(static_cast<int>(g_bno_addr), static_cast<int>(len), static_cast<int>(true));
  if (read_len != len) {
    return false;
  }
  for (size_t i = 0; i < len; ++i) {
    out[i] = Wire.read();
  }
  return true;
}

bool read8(uint8_t reg, uint8_t& value) {
  uint8_t buf[1] = {0};
  if (!readLen(reg, buf, sizeof(buf))) {
    return false;
  }
  value = buf[0];
  return true;
}

int16_t le16(const uint8_t* ptr) {
  return static_cast<int16_t>(
      static_cast<uint16_t>(ptr[0]) |
      (static_cast<uint16_t>(ptr[1]) << 8));
}

bool detectAddress() {
  for (uint8_t addr : {kBnoAddrA, kBnoAddrB}) {
    g_bno_addr = addr;
    uint8_t chip_id = 0;
    if (read8(kRegChipId, chip_id) && chip_id == kChipId) {
      return true;
    }
  }
  g_bno_addr = 0;
  return false;
}

bool setMode(uint8_t mode) {
  if (!write8(kRegOprMode, kModeConfig)) {
    return false;
  }
  delay(25);
  if (!write8(kRegOprMode, mode)) {
    return false;
  }
  delay(25);
  return true;
}

bool softResetBno055() {
  if (!write8(kRegSysTrigger, 0x20)) {
    return false;
  }
  delay(750);
  return detectAddress();
}

bool hasAnyMotionData() {
  uint8_t buf[18] = {0};
  for (int attempt = 0; attempt < 20; ++attempt) {
    if (readLen(kRegAccXLSB, buf, sizeof(buf))) {
      bool non_zero = false;
      for (uint8_t byte_value : buf) {
        if (byte_value != 0) {
          non_zero = true;
          break;
        }
      }
      if (non_zero) {
        return true;
      }
    }
    delay(50);
  }
  return false;
}

bool configureBno055() {
  delay(700);
  if (!detectAddress()) {
    return false;
  }
  if (!write8(kRegPageId, 0x00)) {
    return false;
  }
  if (!softResetBno055()) {
    return false;
  }
  if (!write8(kRegPageId, 0x00)) {
    return false;
  }
  if (!write8(kRegOprMode, kModeConfig)) {
    return false;
  }
  delay(25);
  if (!write8(kRegPwrMode, kPowerNormal)) {
    return false;
  }
  delay(10);
  if (!write8(kRegPageId, 0x00)) {
    return false;
  }
  if (!write8(kRegSysTrigger, 0x00)) {
    return false;
  }
  if (!write8(kRegUnitSel, 0x00)) {
    return false;
  }
  if (!setMode(kModeAmg)) {
    return false;
  }
  return hasAnyMotionData();
}

void printReading(const uint8_t* buf) {
  const float ax = static_cast<float>(le16(buf + 0)) / 100.0f;
  const float ay = static_cast<float>(le16(buf + 2)) / 100.0f;
  const float az = static_cast<float>(le16(buf + 4)) / 100.0f;
  const float mx = static_cast<float>(le16(buf + 6)) / 16.0f;
  const float my = static_cast<float>(le16(buf + 8)) / 16.0f;
  const float mz = static_cast<float>(le16(buf + 10)) / 16.0f;
  const float gx = static_cast<float>(le16(buf + 12)) / 16.0f;
  const float gy = static_cast<float>(le16(buf + 14)) / 16.0f;
  const float gz = static_cast<float>(le16(buf + 16)) / 16.0f;

  Serial.printf(
      "ax=%.3f, ay=%.3f, az=%.3f, gx=%.3f, gy=%.3f, gz=%.3f, mx=%.3f, my=%.3f, mz=%.3f\n",
      ax,
      ay,
      az,
      gx,
      gy,
      gz,
      mx,
      my,
      mz);
}

}  // namespace

void setup() {
  Serial.begin(kBaudRate);
  delay(300);
  Wire.begin(SDA, SCL);
  Wire.setClock(400000);

  g_bno_ready = configureBno055();
  if (!g_bno_ready) {
    Serial.println("bno055_init_failed");
    return;
  }

  Serial.printf("bno055_ready addr=0x%02X mode=AMG sda=%u scl=%u\n", g_bno_addr, SDA, SCL);
}

void loop() {
  if (!g_bno_ready) {
    delay(1000);
    return;
  }

  uint8_t buf[18] = {0};
  if (!readLen(kRegAccXLSB, buf, sizeof(buf))) {
    Serial.println("bno055_read_failed");
    delay(100);
    return;
  }

  printReading(buf);
  delay(kSampleIntervalMs);
}
