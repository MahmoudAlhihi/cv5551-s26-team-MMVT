// MAX9814 Electret Mic Amp - 20kHz Frequency Detector
// Uses three parallel Goertzel filters: 20 kHz target + 16/24 kHz references for SNR.
// Wiring: MAX9814 OUT -> A0, VDD -> 3.3V (or 5V), GND -> GND

#define MIC_PIN      A0
#define NUM_SAMPLES  128
#define SAMPLE_RATE  50000.0   // Hz — target rate (verify at startup)
#define TARGET_FREQ  20000.0   // Hz
#define REF_LOW_FREQ 16000.0   // Hz — noise reference
#define REF_HIGH_FREQ 24000.0  // Hz — noise reference

// Goertzel coefficients — computed once at startup
float coeff_target, coeff_low, coeff_high;
float k_target, k_low, k_high;

void setup() {
  Serial.begin(115200);

  // ADC prescaler = 16 → 1 MHz ADC clock → ~77 kHz max sample rate
  // This is the critical change that makes 20 kHz detection possible.
  ADCSRA = (ADCSRA & 0xF8) | 0x04;

  // Compute Goertzel coefficients for all three bins
  k_target = round((float)NUM_SAMPLES * TARGET_FREQ   / SAMPLE_RATE);
  k_low    = round((float)NUM_SAMPLES * REF_LOW_FREQ  / SAMPLE_RATE);
  k_high   = round((float)NUM_SAMPLES * REF_HIGH_FREQ / SAMPLE_RATE);

  coeff_target = 2.0 * cos(2.0 * PI * k_target / (float)NUM_SAMPLES);
  coeff_low    = 2.0 * cos(2.0 * PI * k_low    / (float)NUM_SAMPLES);
  coeff_high   = 2.0 * cos(2.0 * PI * k_high   / (float)NUM_SAMPLES);

  // Print actual detected frequencies so user can verify alignment
  Serial.print("# target=");
  Serial.print(k_target * SAMPLE_RATE / NUM_SAMPLES, 1);
  Serial.print(" Hz, ref_low=");
  Serial.print(k_low * SAMPLE_RATE / NUM_SAMPLES, 1);
  Serial.print(" Hz, ref_high=");
  Serial.print(k_high * SAMPLE_RATE / NUM_SAMPLES, 1);
  Serial.println(" Hz");

  // Measure actual sample rate (no delay — let analogRead free-run)
  unsigned long t0 = micros();
  for (int i = 0; i < 1000; i++) (void)analogRead(MIC_PIN);
  unsigned long t1 = micros();
  float actual_fs = 1e6f * 1000.0f / (t1 - t0);
  Serial.print("# Free-running fs=");
  Serial.print(actual_fs, 1);
  Serial.println(" Hz");
  Serial.println("# Output format: magnitude,snr,fs_actual");
}

void loop() {
  float q0a=0, q1a=0, q2a=0;  // 20 kHz target
  float q0b=0, q1b=0, q2b=0;  // 16 kHz reference
  float q0c=0, q1c=0, q2c=0;  // 24 kHz reference

  unsigned long t_start = micros();
  for (int i = 0; i < NUM_SAMPLES; i++) {
    float sample = (float)analogRead(MIC_PIN) - 512.0;
    q0a = coeff_target * q1a - q2a + sample;  q2a = q1a; q1a = q0a;
    q0b = coeff_low    * q1b - q2b + sample;  q2b = q1b; q1b = q0b;
    q0c = coeff_high   * q1c - q2c + sample;  q2c = q1c; q1c = q0c;
    // No delayMicroseconds — analogRead at prescaler 16 paces the loop
  }
  unsigned long elapsed = micros() - t_start;
  float fs_actual = 1e6f * NUM_SAMPLES / elapsed;

  float mag_target = sqrt(q1a*q1a + q2a*q2a - q1a*q2a*coeff_target);
  float mag_low    = sqrt(q1b*q1b + q2b*q2b - q1b*q2b*coeff_low);
  float mag_high   = sqrt(q1c*q1c + q2c*q2c - q1c*q2c*coeff_high);

  float snr = mag_target / (0.5f * (mag_low + mag_high) + 1.0f);

  Serial.print(mag_target, 2);
  Serial.print(",");
  Serial.print(snr, 2);
  Serial.print(",");
  Serial.println(fs_actual, 0);
}