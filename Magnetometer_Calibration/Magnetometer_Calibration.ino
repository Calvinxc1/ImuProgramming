#include "MPU9250.h"

// an MPU9250 object with the MPU-9250 sensor on SPI bus 0 and chip select pin 10
MPU9250 IMU(SPI,10);
int status;
const byte interruptPin = 2;

void setup() {
  // serial to display data
  Serial.begin(115200);
  while(!Serial) {}

  // start communication with IMU 
  status = IMU.begin();
  if (status < 0) {
    Serial.println("IMU initialization unsuccessful");
    Serial.println("Check IMU wiring or try cycling power");
    Serial.print("Status: ");
    Serial.println(status);
    while(1) {}
  }

  Serial.println("Initialized");

  // setting DLPF bandwidth to 20 Hz
  IMU.setDlpfBandwidth(MPU9250::DLPF_BANDWIDTH_20HZ);
  // setting SRD to 19 for a 50 Hz update rate
  IMU.setSrd(19);
  // enabling the data ready interrupt
  IMU.enableDataReadyInterrupt();
  // attaching the interrupt to microcontroller pin 2
  pinMode(interruptPin,INPUT);
  attachInterrupt(digitalPinToInterrupt(interruptPin),getIMU,RISING);
}

void loop() {}

void getIMU() {
  // read the sensor
  IMU.readSensor();
  // display the data
  Serial.print(millis());
  Serial.print(",");
  Serial.print(IMU.getMagX_uT(),6);
  Serial.print(",");
  Serial.print(IMU.getMagY_uT(),6);
  Serial.print(",");
  Serial.print(IMU.getMagZ_uT(),6);
  Serial.println();
}
