/* XIAO ESP32S3 Sense - Keyword Spotting with Edge Impulse */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_pdm.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

static const char *TAG = "motion_KWS";

/* Microphone config */
#define SAMPLE_RATE     16000
#define PDM_CLK_GPIO    GPIO_NUM_42
#define PDM_DATA_GPIO   GPIO_NUM_41
#define READ_BUF_SIZE   512

/* LED */
#define LED_PIN         GPIO_NUM_21

static i2s_chan_handle_t rx_chan;
static int16_t *inference_buffer = NULL;

/**
 * Callback for Edge Impulse - provides audio data as float
 */
static int get_audio_signal_data(size_t offset, size_t length, float *out_ptr)
{
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = (float)inference_buffer[offset + i];
    }
    return 0;
}

/**
 * Initialize PDM microphone using new I2S driver
 */
void init_microphone(void)
{
    
    // Creates an I2S channel on port 0 (This board has 2 in total but we only need one).
    // I2S_ROLE_MASTER means the ESP32S3 generates the clock signal for the microphone - the mircophone is the slave which must listen to this clock.
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

    // The i2s_pdm_rx_config_t structure is used to configure the PDM microphone settings for the I2S driver. 
    i2s_pdm_rx_config_t pdm_rx_cfg = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(SAMPLE_RATE), // This macro sets up the clock timing to achieve a 16kHz output sample rate.
        // Configures the audio format. I2S_DATA_BIT_WIDTH_16BIT means each sample is stored as a 16-bit signed integer and given 65,536 possible amplitude levels.
        // I2S_SLOT_MODE_MONO meeans we are reading from one microphone.
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO), 
        .gpio_cfg = {
            .clk = PDM_CLK_GPIO,
            .din = PDM_DATA_GPIO,
            .invert_flags = {
                .clk_inv = false,
            },
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx_chan, &pdm_rx_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
    ESP_LOGI(TAG, "Microphone initialized (PDM on GPIO %d/%d)", PDM_CLK_GPIO, PDM_DATA_GPIO);
}

/**
 * Print inference results
 */
void print_inference_result(ei_impulse_result_t result)
{
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
        result.timing.dsp,
        result.timing.classification,
        result.timing.anomaly);

    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: %.5f\r\n",
            ei_classifier_inferencing_categories[i],
            result.classification[i].value);
    }

#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif
}

/* Main application entry point */

// Using extern "C" because this is a C++ file but app_main needs C linkage for ESP-IDF. Without it the C++ compiler 
// would mangle the function name and the startip code wouldn't find it.
extern "C" int app_main() 
{
    /* Setup LED */
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(LED_PIN, 1); // LED off (active low)

    /* Init microphone */
    init_microphone();

    /* Print memory info */
    ESP_LOGI(TAG, "Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());
    ESP_LOGI(TAG, "Free PSRAM: %lu bytes",
        (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "Model expects %d samples (%d ms window)",
        EI_CLASSIFIER_RAW_SAMPLE_COUNT,
        EI_CLASSIFIER_RAW_SAMPLE_COUNT / (SAMPLE_RATE / 1000));

    /* Allocate inference buffer */
    // The inference buffer holds one full window of audio (16kmHz * 1s = 16000 samples) as int16_t. We try to allocate this in PSRAM if available, since it is large (32KB), but fall back to regular heap if not.
    int buffer_size = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    // heaps_caps_malloc is used to specify the type of memory to allocate from. This method means we are allocating to the heap with the MALLOC_CAP_SPIRAM capability, which is PSRAM on ESP32. If PSRAM is not available or allocation fails, it will return NULL.
    // The caps refer to the capabilities of the memory being allocated. MALLOC_CAP_SPIRAM means we want to allocate from PSRAM, which is external RAM that is larger than the internal RAM but has higher latency. This is ideal for large buffers that don't require fast access, like our inference buffer.
    inference_buffer = (int16_t *)heap_caps_malloc(
        buffer_size * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (inference_buffer == NULL) {
        /* Fall back to regular malloc if no PSRAM */
        inference_buffer = (int16_t *)malloc(buffer_size * sizeof(int16_t));
    }
    if (inference_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate inference buffer");
        return 1;
    }

    /* Allocate read buffer on heap */
    // The read buffer is smaller (512 samples) and is used for reading chunks of audio from the microphone before copying them into the inference buffer. We can allocate this on regular heap.
    int16_t *read_buf = (int16_t *)malloc(READ_BUF_SIZE * sizeof(int16_t));
    if (read_buf == NULL) {
        ESP_LOGE(TAG, "Failed to allocate read buffer");
        return 1;
    }

    ei_impulse_result_t result = {};
    size_t bytes_read;

    ESP_LOGI(TAG, "Starting keyword spotting loop...");

    while (true) {
        /* Step 1: Fill inference buffer with one window of audio */
        // Reads 512 samples at a time from the microphone (through DMA/I2S) from read buffer and copies them into the inference buffer.
        // It leeps going until the buffer is full.
        int ix = 0;
        while (ix < buffer_size) {
            int samples_needed = buffer_size - ix;
            int to_read = (samples_needed < READ_BUF_SIZE) ? samples_needed : READ_BUF_SIZE;

            esp_err_t ret = i2s_channel_read(rx_chan, read_buf,
                to_read * sizeof(int16_t), &bytes_read, pdMS_TO_TICKS(1000));

            if (ret == ESP_OK && bytes_read > 0) {
                int samples_read = bytes_read / sizeof(int16_t);
                memcpy(&inference_buffer[ix], read_buf,
                    samples_read * sizeof(int16_t));
                ix += samples_read;
            }
        }

        /* Step 2: Set up the signal */
        // Creates the signal_t structure that Edge Impulse uses to access the audio data.
        // It tells the classifier how many samples are available and which callback function to use to retrieve them.
        signal_t signal;
        signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
        signal.get_data = &get_audio_signal_data;

        /* Step 3: Run inference - LED on while processing */
        // run_classifier does everything in one call 
        // - it runs the MFCC feature extraction on the raw audio data
        // - it feeds the resulting features into the quantized TFLite Micro NN
        // - it optionally runs the anomoly detection post-processing
        gpio_set_level(LED_PIN, 0); // LED on (active low)

        EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);

        gpio_set_level(LED_PIN, 1); // LED off

        if (err != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", err);
            continue;
        }

        /* Step 4: Print results */
        print_inference_result(result);

        /* Step 5: Check for keyword detection */
        for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            if (result.classification[i].value > 0.8) { // result.classification contains an array of confidence scores - one for each label in your model.
                ESP_LOGW(TAG, ">>> Detected: %s (%.1f%%) <<<",
                    ei_classifier_inferencing_categories[i],
                    result.classification[i].value * 100);
            }
        }
    }
}