"""Exact paper table values, kept separate from measured experiment results."""

DEFENSES = ["Raw", "CE-AT", "DT-AT", "DMW", "TI", "LPF", "WaveGuard", "AudioPure", "PVP Vote"]
TABLE_1 = {
    "en": {
        "Clean": [4.75, 5.72, 5.70, 5.86, 5.81, 6.45, 6.76, 6.17, 4.80],
        "PGD": [61.06, 8.69, 8.13, 9.61, 9.80, 16.28, 15.47, 8.38, 45.62],
        "MI-FGSM": [77.65, 16.20, 16.98, 16.86, 16.85, 39.24, 32.25, 27.33, 76.11],
        "VMI-FGSM": [78.84, 30.29, 30.91, 34.48, 39.14, 42.04, 55.36, 42.38, 78.34],
        "Muting Whisper": [99.52, 6.11, 6.09, 5.94, 6.04, 7.52, 7.62, 6.80, 79.52],
        "Sloth": [39.50, 9.38, 8.71, 9.26, 9.38, 14.88, 14.98, 8.70, 28.47],
        "Feature attack": [75.43, 71.26, 67.23, 70.61, 70.63, 68.39, 57.03, 70.86, 78.60],
    },
    "zh": {
        "Clean": [6.29, 5.51, 5.87, 5.66, 5.22, 10.03, 7.74, 27.67, 5.75],
        "PGD": [69.01, 15.08, 14.86, 15.19, 17.39, 26.30, 12.67, 26.71, 64.27],
        "MI-FGSM": [79.87, 23.29, 23.72, 28.33, 23.25, 51.71, 59.45, 33.49, 74.18],
        "VMI-FGSM": [80.36, 31.26, 31.90, 35.58, 40.39, 43.38, 57.13, 43.73, 79.82],
        "Muting Whisper": [94.47, 5.93, 6.05, 5.87, 5.41, 6.74, 6.89, 5.96, 75.24],
        "Sloth": [53.11, 14.33, 11.26, 8.43, 9.82, 36.04, 31.92, 20.24, 54.29],
        "Feature attack": [72.25, 66.57, 66.10, 66.07, 65.49, 72.10, 70.34, 69.92, 75.31],
    },
}
TABLE_4 = {
    "Clean": (3.075, 3.714, 3.755),
    "PGD": (2.761, 3.094, 3.314),
    "MI-FGSM": (2.804, 2.523, 2.518),
    "VMI-FGSM": (2.763, 2.517, 2.528),
    "Muting Whisper": (3.011, 3.640, 3.729),
    "Sloth": (2.895, 3.152, 3.458),
    "Feature attack": (2.954, 3.526, 3.590),
}
OTHER_PROSE_ANCHORS = {
    "whisper_tiny_DT-AT_WER": 64.90, "whisper_tiny_WaveGuard_WER": 54.75,
    "whisper_large_raw_WER": 52.96, "hubert_raw_WER": 39.80,
    "wav2vec2_raw_WER": 42.12, "wav2vec2_LPF_WER": 48.24,
    "without_attack_loss_WER": 6.6, "mel_correlation_one_unidentified_sample": 0.9733,
    "clean_OTA_WER": 7.45, "adversarial_OTA_WER": 78.23,
}


def compare_table_1(summary, language, condition, defense="Raw", variant="adversarial"):
    reference = TABLE_1[language][condition][DEFENSES.index(defense)]
    measured = summary["summary"][variant]["error_percent"]
    return {"language": language, "condition": condition, "defense": defense,
            "variant": variant, "metric": "CER" if language == "zh" else "WER",
            "paper_percent": reference, "observed_percent": measured,
            "difference_percentage_points": measured - reference,
            "warning": "Paper artifacts/manifest/decoder may differ; a numeric gap does not establish a code bug."}
