# Game Realtime OCR Camera — Android Prototype

Test APK prototype for real-time camera OCR and translation.

- CameraX live back camera.
- Bundled ML Kit OCR for Latin, Japanese, Korean, and Simplified Chinese.
- ML Kit on-device translation with automatic language-model download.
- Original text region is covered with a dark overlay and translated text is drawn in the same position.
- OCR is throttled and translations are cached to reduce CPU/battery use.
- Model menu can download/check, delete the selected target model, and show downloaded-model count.

This first prototype uses a dark cover rectangle, not scene-aware inpainting yet.
