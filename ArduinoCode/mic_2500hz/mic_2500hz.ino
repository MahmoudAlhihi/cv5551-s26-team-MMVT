// MAX9814 Electret Mic Amp - 2.5kHz Frequency Detector
// Uses the Goertzel algorithm to measure signal power at a single target frequency.
// Wiring: MAX9814 OUT -> A0, VDD -> 3.3V (or 5V), GND -> GND
//         (GAIN pin left floating = 60dB gain; pull to GND = 50dB, pull to VCC = 40dB)

#define MIC_PIN      A0
#define NUM_SAMPLES  128
#define SAMPLE_RATE  9615.0   // Hz — analogRead takes ~104 us, giving ~9615 Hz
#define TARGET_FREQ  2500.0   // Hz

// Goertzel coefficients — computed once at startup
float goertzel_coeff;
float goertzel_k;

void setup() {
  Serial.begin(115200);

  // k is the DFT bin closest to the target frequency
  goertzel_k     = round((float)NUM_SAMPLES * TARGET_FREQ / SAMPLE_RATE);
  float omega    = 2.0 * PI * goertzel_k / (float)NUM_SAMPLES;
  goertzel_coeff = 2.0 * cos(omega);

  // Print the actual detected frequency so the user can verify alignment
  float actual_freq = goertzel_k * SAMPLE_RATE / NUM_SAMPLES;
  Serial.print("# Goertzel k=");
  Serial.print((int)goertzel_k);
  Serial.print("  actual_freq=");
  Serial.print(actual_freq, 1);
  Serial.println(" Hz");
}

void loop() {
  float q0 = 0.0, q1 = 0.0, q2 = 0.0;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    // Center the 0-1023 ADC reading around zero
    float sample = (float)analogRead(MIC_PIN) - 512.0;

    q0 = goertzel_coeff * q1 - q2 + sample;
    q2 = q1;
    q1 = q0;

    delayMicroseconds(104);  // maintain ~9615 Hz sample rate
  }

  // Goertzel magnitude (square root of power)
  float magnitude = sqrt(q1 * q1 + q2 * q2 - q1 * q2 * goertzel_coeff);

  Serial.println(magnitude, 2);
}