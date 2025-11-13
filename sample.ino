#include <Wire.h>

#define ADDR1 0x68
#define ADDR2 0x69
uint8_t dev = ADDR1;

void readWHO() {
  Wire.beginTransmission(dev);
  Wire.write(0x75); // WHO_AM_I
  Wire.endTransmission(false);
  Wire.requestFrom(dev, (uint8_t)1);
  if (Wire.available()) {
    uint8_t who = Wire.read();
    Serial.print("WHO_AM_I(0x"); Serial.print(dev, HEX); Serial.print(") = 0x");
    Serial.println(who, HEX);
  } else {
    Serial.print("No WHO_AM_I from 0x"); Serial.println(dev, HEX);
  }
}

bool wakeDevice() {
  Wire.beginTransmission(dev);
  Wire.write(0x6B); // PWR_MGMT_1
  Wire.write(0x00); // wake
  if (Wire.endTransmission() == 0) return true;
  return false;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Wire.begin(21, 22);
  delay(200);
  
  dev = ADDR1;
  wakeDevice();
  readWHO();
  delay(200);

  dev = ADDR2;
  wakeDevice();
  readWHO();

  dev = ADDR1; // use default
}

int16_t read16(uint8_t reg) {
  Wire.beginTransmission(dev);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(dev, (uint8_t)2, (uint8_t)true);
  int16_t hi = Wire.read();
  int16_t lo = Wire.read();
  return (hi << 8) | lo;
}

void loop() {
  // --- 加速度データ ---
  int16_t ax = read16(0x3B);
  int16_t ay = read16(0x3D);
  int16_t az = read16(0x3F);

  // --- ジャイロデータ ---
  int16_t gx = read16(0x43);
  int16_t gy = read16(0x45);
  int16_t gz = read16(0x47);

  // --- 単位変換 ---
  const float accelScale = 9.80665 / 16384.0;   // m/s^2 (±2g)
  const float gyroScale = 1.0 / 131.0;          // deg/s (±250°/s)

  float aX = ax * accelScale;
  float aY = ay * accelScale;
  float aZ = az * accelScale;

  float gX = gx * gyroScale;
  float gY = gy * gyroScale;
  float gZ = gz * gyroScale;

  // --- シリアル出力（Pythonがパースしやすい形式） ---
  Serial.print("ax: "); Serial.print(aX, 3);
  Serial.print(" ay: "); Serial.print(aY, 3);
  Serial.print(" az: "); Serial.print(aZ, 3);
  Serial.print(" gx: "); Serial.print(gX, 3);
  Serial.print(" gy: "); Serial.print(gY, 3);
  Serial.print(" gz: "); Serial.println(gZ, 3);

  delay(100); // 更新頻度を調整
}
