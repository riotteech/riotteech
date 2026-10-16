package com.riotteech.gamerealtimeocr;

import android.Manifest;
import android.app.AlertDialog;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Rect;
import android.graphics.RectF;
import android.media.Image;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Size;
import android.view.Gravity;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.common.model.DownloadConditions;
import com.google.mlkit.common.model.RemoteModelManager;
import com.google.mlkit.nl.translate.TranslateLanguage;
import com.google.mlkit.nl.translate.TranslateRemoteModel;
import com.google.mlkit.nl.translate.Translation;
import com.google.mlkit.nl.translate.Translator;
import com.google.mlkit.nl.translate.TranslatorOptions;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions;
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions;
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

public class MainActivity extends AppCompatActivity {
    private static final int CAMERA_PERMISSION_REQUEST = 1001;

    // Faster re-detection than v0.1. Backpressure still guarantees that slow
    // devices never build an OCR queue.
    private static final long OCR_INTERVAL_MS = 160L;

    // A translated label is not deleted on a single OCR miss. This is the
    // main anti-flicker window for hand-held camera movement.
    private static final long TRACK_HOLD_MS = 1250L;
    private static final long TRACK_SOFT_PREDICT_MS = 520L;
    private static final int MAX_TRACKS = 48;

    private final LinkedHashMap<String, String> translateLanguages = new LinkedHashMap<>();

    private final Map<String, String> translationCache = Collections.synchronizedMap(
            new LinkedHashMap<String, String>(256, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<String, String> eldest) {
                    return size() > 700;
                }
            });

    private final Set<String> translationPending =
            Collections.synchronizedSet(new HashSet<>());

    private final Object trackLock = new Object();
    private final List<TextTrack> tracks = new ArrayList<>();
    private final AtomicLong nextTrackId = new AtomicLong(1);

    private PreviewView previewView;
    private TranslationOverlayView overlayView;
    private Spinner sourceSpinner;
    private Spinner targetSpinner;
    private TextView statusView;
    private Button modelButton;

    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;
    private TextRecognizer recognizer;
    private Translator translator;

    private final AtomicBoolean ocrBusy = new AtomicBoolean(false);
    private final AtomicLong frameGeneration = new AtomicLong(0);

    private volatile long lastOcrAt = 0L;
    private volatile int currentFrameWidth = 1;
    private volatile int currentFrameHeight = 1;

    private static class Detection {
        final Rect box;
        final String source;
        final String key;

        Detection(Rect box, String source, String key) {
            this.box = new Rect(box);
            this.source = source;
            this.key = key;
        }
    }

    private static class TextTrack {
        final long id;
        final RectF box = new RectF();

        String source;
        String key;
        String translated;

        long lastSeenAt;
        int missedFrames;
        boolean matchedThisFrame;

        // Movement between OCR observations, in image-coordinate pixels per
        // OCR update. Used briefly when OCR misses during camera motion.
        float velocityX;
        float velocityY;

        TextTrack(long id, Detection detection, String translated, long now) {
            this.id = id;
            this.box.set(detection.box);
            this.source = detection.source;
            this.key = detection.key;
            this.translated = translated;
            this.lastSeenAt = now;
            this.missedFrames = 0;
            this.matchedThisFrame = true;
        }

        float centerX() {
            return box.centerX();
        }

        float centerY() {
            return box.centerY();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setupLanguages();
        buildUi();

        cameraExecutor = Executors.newSingleThreadExecutor();

        configureTargetSpinner();
        rebuildRecognizer();
        rebuildTranslator(true);

        sourceSpinner.setOnItemSelectedListener(
                new SimpleItemSelectedListener(() -> {
                    configureTargetSpinner();
                    clearTrackingState();
                    rebuildRecognizer();
                    rebuildTranslator(true);
                    overlayView.clearItems();
                }));

        targetSpinner.setOnItemSelectedListener(
                new SimpleItemSelectedListener(() -> {
                    clearTrackingState();
                    rebuildTranslator(true);
                    overlayView.clearItems();
                }));

        modelButton.setOnClickListener(v -> showModelDialog());

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_REQUEST);
        }
    }

    private void setupLanguages() {
        translateLanguages.put("English", TranslateLanguage.ENGLISH);
        translateLanguages.put("Türkçe", TranslateLanguage.TURKISH);
        translateLanguages.put("Deutsch", TranslateLanguage.GERMAN);
        translateLanguages.put("Français", TranslateLanguage.FRENCH);
        translateLanguages.put("Español", TranslateLanguage.SPANISH);
        translateLanguages.put("Italiano", TranslateLanguage.ITALIAN);
        translateLanguages.put("Português", TranslateLanguage.PORTUGUESE);
        translateLanguages.put("日本語", TranslateLanguage.JAPANESE);
        translateLanguages.put("한국어", TranslateLanguage.KOREAN);
        translateLanguages.put("简体中文", TranslateLanguage.CHINESE);
        translateLanguages.put("Русский", TranslateLanguage.RUSSIAN);
        translateLanguages.put("العربية", TranslateLanguage.ARABIC);
    }

    private void buildUi() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.BLACK);

        previewView = new PreviewView(this);
        previewView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);

        root.addView(
                previewView,
                new FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.MATCH_PARENT));

        overlayView = new TranslationOverlayView(this);
        overlayView.setClickable(false);

        root.addView(
                overlayView,
                new FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.MATCH_PARENT));

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(10), dp(8), dp(10), dp(8));
        panel.setBackgroundColor(Color.argb(215, 18, 18, 18));

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);

        sourceSpinner = new Spinner(this);
        sourceSpinner.setAdapter(
                new ArrayAdapter<>(
                        this,
                        android.R.layout.simple_spinner_dropdown_item,
                        new ArrayList<>(
                                Arrays.asList(
                                        "English",
                                        "Türkçe",
                                        "Deutsch",
                                        "Français",
                                        "Español",
                                        "Italiano",
                                        "Português",
                                        "日本語",
                                        "한국어",
                                        "简体中文"))));

        row.addView(
                sourceSpinner,
                new LinearLayout.LayoutParams(
                        0,
                        dp(48),
                        1f));

        TextView arrow = new TextView(this);
        arrow.setText("  →  ");
        arrow.setTextColor(Color.WHITE);
        arrow.setTextSize(18);
        row.addView(arrow);

        targetSpinner = new Spinner(this);

        row.addView(
                targetSpinner,
                new LinearLayout.LayoutParams(
                        0,
                        dp(48),
                        1f));

        modelButton = new Button(this);
        modelButton.setText("Model");

        row.addView(
                modelButton,
                new LinearLayout.LayoutParams(
                        dp(86),
                        dp(48)));

        panel.addView(row);

        statusView = new TextView(this);
        statusView.setTextColor(Color.WHITE);
        statusView.setTextSize(13);
        statusView.setText("Hazırlanıyor…");
        statusView.setPadding(dp(4), dp(2), dp(4), dp(2));

        panel.addView(statusView);

        root.addView(
                panel,
                new FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.WRAP_CONTENT,
                        Gravity.TOP));

        setContentView(root);
    }

    private int dp(int value) {
        return Math.round(
                value
                        * getResources()
                        .getDisplayMetrics()
                        .density);
    }

    private String selectedSourceName() {
        Object item = sourceSpinner.getSelectedItem();
        return item == null
                ? "English"
                : item.toString();
    }

    private String selectedTargetName() {
        Object item = targetSpinner.getSelectedItem();
        return item == null
                ? "Türkçe"
                : item.toString();
    }

    private String selectedSourceCode() {
        return translateLanguages.get(
                selectedSourceName());
    }

    private String selectedTargetCode() {
        return translateLanguages.get(
                selectedTargetName());
    }

    private String currentPairPrefix() {
        return selectedSourceCode()
                + ">"
                + selectedTargetCode()
                + ">";
    }

    private void configureTargetSpinner() {
        String source = selectedSourceName();
        String previous = selectedTargetName();

        List<String> targets = new ArrayList<>();

        for (String language : translateLanguages.keySet()) {
            if (!language.equals(source)) {
                targets.add(language);
            }
        }

        targetSpinner.setAdapter(
                new ArrayAdapter<>(
                        this,
                        android.R.layout.simple_spinner_dropdown_item,
                        targets));

        int preferred = targets.indexOf(previous);

        if (preferred < 0) {
            preferred = targets.indexOf("Türkçe");
        }

        if (preferred < 0) {
            preferred = 0;
        }

        targetSpinner.setSelection(
                preferred,
                false);
    }

    private synchronized void rebuildRecognizer() {
        if (recognizer != null) {
            recognizer.close();
        }

        String source = selectedSourceName();

        if ("日本語".equals(source)) {
            recognizer = TextRecognition.getClient(
                    new JapaneseTextRecognizerOptions.Builder().build());
        } else if ("한국어".equals(source)) {
            recognizer = TextRecognition.getClient(
                    new KoreanTextRecognizerOptions.Builder().build());
        } else if ("简体中文".equals(source)) {
            recognizer = TextRecognition.getClient(
                    new ChineseTextRecognizerOptions.Builder().build());
        } else {
            recognizer = TextRecognition.getClient(
                    TextRecognizerOptions.DEFAULT_OPTIONS);
        }

        translationCache.clear();
        translationPending.clear();
    }

    private synchronized void rebuildTranslator(
            boolean downloadIfNeeded) {
        if (translator != null) {
            translator.close();
            translator = null;
        }

        String sourceCode = selectedSourceCode();
        String targetCode = selectedTargetCode();

        if (sourceCode == null
                || targetCode == null
                || sourceCode.equals(targetCode)) {
            setStatus(
                    "Kaynak ve hedef dili farklı seçin.");
            return;
        }

        TranslatorOptions options =
                new TranslatorOptions.Builder()
                        .setSourceLanguage(sourceCode)
                        .setTargetLanguage(targetCode)
                        .build();

        translator = Translation.getClient(options);

        translationCache.clear();
        translationPending.clear();

        if (downloadIfNeeded) {
            setStatus(
                    "Çeviri modeli kontrol ediliyor…");

            translator
                    .downloadModelIfNeeded(
                            new DownloadConditions.Builder().build())
                    .addOnSuccessListener(
                            unused -> setStatus(
                                    selectedSourceName()
                                            + " → "
                                            + selectedTargetName()
                                            + " modeli hazır • stabil takip aktif"))
                    .addOnFailureListener(
                            e -> setStatus(
                                    "Model indirilemedi: "
                                            + e.getMessage()));
        }
    }

    private void clearTrackingState() {
        synchronized (trackLock) {
            tracks.clear();
        }

        frameGeneration.incrementAndGet();
        translationPending.clear();

        if (overlayView != null) {
            overlayView.clearItems();
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(this);

        future.addListener(
                () -> {
                    try {
                        cameraProvider = future.get();
                        bindCameraUseCases();
                    } catch (Exception e) {
                        setStatus(
                                "Kamera açılamadı: "
                                        + e.getMessage());
                    }
                },
                ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        if (cameraProvider == null) {
            return;
        }

        cameraProvider.unbindAll();

        Preview preview =
                new Preview.Builder().build();

        preview.setSurfaceProvider(
                previewView.getSurfaceProvider());

        ImageAnalysis analysis =
                new ImageAnalysis.Builder()
                        .setTargetResolution(
                                new Size(1280, 720))
                        .setBackpressureStrategy(
                                ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        analysis.setAnalyzer(
                cameraExecutor,
                this::analyzeFrame);

        try {
            cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis);

            setStatus(
                    "Kamera hazır • stabil metin takibi aktif");
        } catch (Exception e) {
            setStatus(
                    "Kamera bağlanamadı: "
                            + e.getMessage());
        }
    }

    @android.annotation.SuppressLint(
            "UnsafeOptInUsageError")
    private void analyzeFrame(
            @NonNull ImageProxy imageProxy) {
        long now = SystemClock.elapsedRealtime();

        if (now - lastOcrAt < OCR_INTERVAL_MS
                || !ocrBusy.compareAndSet(
                        false,
                        true)) {
            imageProxy.close();
            return;
        }

        lastOcrAt = now;

        Image mediaImage = imageProxy.getImage();
        TextRecognizer localRecognizer = recognizer;

        if (mediaImage == null
                || localRecognizer == null) {
            ocrBusy.set(false);
            imageProxy.close();
            return;
        }

        int rotation =
                imageProxy
                        .getImageInfo()
                        .getRotationDegrees();

        final int uprightWidth =
                (rotation == 90 || rotation == 270)
                        ? imageProxy.getHeight()
                        : imageProxy.getWidth();

        final int uprightHeight =
                (rotation == 90 || rotation == 270)
                        ? imageProxy.getWidth()
                        : imageProxy.getHeight();

        InputImage image =
                InputImage.fromMediaImage(
                        mediaImage,
                        rotation);

        long generation =
                frameGeneration.incrementAndGet();

        localRecognizer
                .process(image)
                .addOnSuccessListener(
                        result -> handleRecognizedText(
                                generation,
                                result,
                                uprightWidth,
                                uprightHeight))
                .addOnFailureListener(
                        e -> {
                            keepTracksAliveOnMiss();
                            setStatus(
                                    "OCR geçici hata • takip korunuyor: "
                                            + e.getMessage());
                        })
                .addOnCompleteListener(
                        task -> {
                            ocrBusy.set(false);
                            imageProxy.close();
                        });
    }

    private void handleRecognizedText(
            long generation,
            Text text,
            int frameWidth,
            int frameHeight) {
        // Do not reject a valid OCR result merely because a newer camera
        // frame exists. ImageAnalysis already uses KEEP_ONLY_LATEST and
        // ocrBusy allows only one OCR operation at a time.
        List<Detection> detections =
                new ArrayList<>();

        for (Text.TextBlock block :
                text.getTextBlocks()) {
            Rect box = block.getBoundingBox();
            String source =
                    normalizeText(block.getText());

            if (box == null
                    || source.length() < 2
                    || box.width() < 12
                    || box.height() < 10) {
                continue;
            }

            String key =
                    trackingKey(source);

            if (key.length() < 2) {
                continue;
            }

            detections.add(
                    new Detection(
                            box,
                            source,
                            key));
        }

        currentFrameWidth = frameWidth;
        currentFrameHeight = frameHeight;

        long now =
                SystemClock.elapsedRealtime();

        updateTracks(
                detections,
                now);

        refreshOverlay();

        if (detections.isEmpty()) {
            int active = activeTranslatedTrackCount(now);

            if (active > 0) {
                setStatus(
                        "OCR kısa süreli kayıp • "
                                + active
                                + " çeviri konumu korunuyor");
            } else {
                setStatus(
                        "Yazı aranıyor…");
            }

            return;
        }

        List<String> translateNow =
                new ArrayList<>();

        synchronized (trackLock) {
            for (TextTrack track : tracks) {
                if (!track.matchedThisFrame) {
                    continue;
                }

                String cached =
                        translationCache.get(
                                cacheKey(track.source));

                if (cached != null
                        && !cached.isEmpty()) {
                    track.translated = cached;
                } else if (
                        track.translated == null
                                || track.translated.isEmpty()
                                || !isTranslationCompatible(
                                        track.source,
                                        track.key)) {
                    translateNow.add(
                            track.source);
                }
            }
        }

        for (String source : translateNow) {
            translateText(source);
        }

        setStatus(
                "OCR aktif • "
                        + detections.size()
                        + " metin • stabil takip");
    }

    private void updateTracks(
            List<Detection> detections,
            long now) {
        synchronized (trackLock) {
            for (TextTrack track : tracks) {
                track.matchedThisFrame = false;
            }

            boolean[] detectionUsed =
                    new boolean[
                            detections.size()];

            // Strongest matches first. This preserves a translated label even
            // when ML Kit changes punctuation or line wrapping slightly.
            while (true) {
                TextTrack bestTrack = null;
                int bestDetectionIndex = -1;
                float bestScore = 0.0f;

                for (TextTrack track : tracks) {
                    if (track.matchedThisFrame) {
                        continue;
                    }

                    for (int i = 0;
                         i < detections.size();
                         i++) {
                        if (detectionUsed[i]) {
                            continue;
                        }

                        Detection detection =
                                detections.get(i);

                        float score =
                                trackMatchScore(
                                        track,
                                        detection,
                                        currentFrameWidth,
                                        currentFrameHeight);

                        if (score > bestScore) {
                            bestScore = score;
                            bestTrack = track;
                            bestDetectionIndex = i;
                        }
                    }
                }

                if (bestTrack == null
                        || bestDetectionIndex < 0
                        || bestScore < 0.54f) {
                    break;
                }

                Detection detection =
                        detections.get(
                                bestDetectionIndex);

                updateMatchedTrack(
                        bestTrack,
                        detection,
                        now);

                bestTrack.matchedThisFrame = true;
                detectionUsed[
                        bestDetectionIndex] = true;
            }

            // Create tracks for truly new text.
            for (int i = 0;
                 i < detections.size();
                 i++) {
                if (detectionUsed[i]) {
                    continue;
                }

                Detection detection =
                        detections.get(i);

                String cached =
                        translationCache.get(
                                cacheKey(
                                        detection.source));

                tracks.add(
                        new TextTrack(
                                nextTrackId.getAndIncrement(),
                                detection,
                                cached,
                                now));
            }

            // Keep translations over short OCR gaps instead of clearing them.
            Iterator<TextTrack> iterator =
                    tracks.iterator();

            while (iterator.hasNext()) {
                TextTrack track =
                        iterator.next();

                if (track.matchedThisFrame) {
                    track.missedFrames = 0;
                    continue;
                }

                track.missedFrames++;

                if (now - track.lastSeenAt
                        > TRACK_HOLD_MS) {
                    iterator.remove();
                }
            }

            // Defensive bound in text-heavy scenes.
            if (tracks.size() > MAX_TRACKS) {
                tracks.sort(
                        (a, b) ->
                                Long.compare(
                                        b.lastSeenAt,
                                        a.lastSeenAt));

                while (tracks.size() > MAX_TRACKS) {
                    tracks.remove(
                            tracks.size() - 1);
                }
            }
        }
    }

    private void updateMatchedTrack(
            TextTrack track,
            Detection detection,
            long now) {
        float oldCenterX =
                track.box.centerX();

        float oldCenterY =
                track.box.centerY();

        float rawDx =
                detection.box.exactCenterX()
                        - oldCenterX;

        float rawDy =
                detection.box.exactCenterY()
                        - oldCenterY;

        float reference =
                Math.max(
                        22.0f,
                        (float) Math.hypot(
                                track.box.width(),
                                track.box.height()));

        float movement =
                (float) Math.hypot(
                        rawDx,
                        rawDy)
                        / reference;

        // Small movements are smoothed strongly to remove OCR jitter.
        // Large camera movements catch up quickly so the overlay follows
        // the physical text rather than floating behind it.
        float alpha;

        if (movement > 1.15f) {
            alpha = 0.94f;
        } else if (movement > 0.55f) {
            alpha = 0.82f;
        } else if (movement > 0.20f) {
            alpha = 0.68f;
        } else {
            alpha = 0.48f;
        }

        track.velocityX =
                track.velocityX * 0.52f
                        + rawDx * 0.48f;

        track.velocityY =
                track.velocityY * 0.52f
                        + rawDy * 0.48f;

        lerpRect(
                track.box,
                detection.box,
                alpha);

        float textSimilarity =
                textSimilarity(
                        track.key,
                        detection.key);

        // Preserve the already visible translation for small OCR variations.
        // A materially different sentence keeps the track position but asks
        // for a fresh translation.
        if (textSimilarity < 0.82f) {
            String cached =
                    translationCache.get(
                            cacheKey(
                                    detection.source));

            if (cached != null
                    && !cached.isEmpty()) {
                track.translated = cached;
            } else if (textSimilarity < 0.62f) {
                track.translated = null;
            }
        }

        track.source =
                detection.source;

        track.key =
                detection.key;

        track.lastSeenAt = now;
    }

    private void lerpRect(
            RectF current,
            Rect target,
            float alpha) {
        current.left +=
                (target.left - current.left)
                        * alpha;

        current.top +=
                (target.top - current.top)
                        * alpha;

        current.right +=
                (target.right - current.right)
                        * alpha;

        current.bottom +=
                (target.bottom - current.bottom)
                        * alpha;
    }

    private float trackMatchScore(
            TextTrack track,
            Detection detection,
            int frameWidth,
            int frameHeight) {
        float text =
                textSimilarity(
                        track.key,
                        detection.key);

        float overlap =
                iou(
                        track.box,
                        detection.box);

        float distance =
                normalizedCenterDistance(
                        track.box,
                        detection.box,
                        frameWidth,
                        frameHeight);

        float distanceScore =
                Math.max(
                        0.0f,
                        1.0f - distance);

        // Exact/similar text dominates. Geometry is deliberately secondary
        // so the same sign/subtitle can be re-associated after a camera move.
        float score =
                text * 0.76f
                        + overlap * 0.14f
                        + distanceScore * 0.10f;

        if (text >= 0.96f) {
            score += 0.18f;
        } else if (text >= 0.82f) {
            score += 0.08f;
        }

        if (overlap >= 0.35f
                && text >= 0.45f) {
            score += 0.08f;
        }

        return score;
    }

    private float iou(
            RectF a,
            Rect b) {
        float left =
                Math.max(
                        a.left,
                        b.left);

        float top =
                Math.max(
                        a.top,
                        b.top);

        float right =
                Math.min(
                        a.right,
                        b.right);

        float bottom =
                Math.min(
                        a.bottom,
                        b.bottom);

        float intersection =
                Math.max(
                        0.0f,
                        right - left)
                        * Math.max(
                                0.0f,
                                bottom - top);

        if (intersection <= 0.0f) {
            return 0.0f;
        }

        float union =
                a.width()
                        * a.height()
                        + b.width()
                        * b.height()
                        - intersection;

        if (union <= 0.0f) {
            return 0.0f;
        }

        return intersection / union;
    }

    private float normalizedCenterDistance(
            RectF a,
            Rect b,
            int frameWidth,
            int frameHeight) {
        float dx =
                a.centerX()
                        - b.exactCenterX();

        float dy =
                a.centerY()
                        - b.exactCenterY();

        float diagonal =
                (float) Math.hypot(
                        Math.max(
                                1,
                                frameWidth),
                        Math.max(
                                1,
                                frameHeight));

        return Math.min(
                1.5f,
                (float) Math.hypot(
                        dx,
                        dy)
                        / Math.max(
                                1.0f,
                                diagonal * 0.55f));
    }

    private String normalizeText(
            String value) {
        if (value == null) {
            return "";
        }

        return value
                .replaceAll(
                        "\\s+",
                        " ")
                .trim();
    }

    private String trackingKey(
            String value) {
        String normalized =
                Normalizer.normalize(
                                value == null
                                        ? ""
                                        : value,
                                Normalizer.Form.NFKC)
                        .toLowerCase(
                                Locale.ROOT);

        StringBuilder builder =
                new StringBuilder(
                        normalized.length());

        for (int offset = 0;
             offset < normalized.length();) {
            int codePoint =
                    normalized.codePointAt(
                            offset);

            if (Character.isLetterOrDigit(
                    codePoint)) {
                builder.appendCodePoint(
                        codePoint);
            }

            offset +=
                    Character.charCount(
                            codePoint);
        }

        return builder.toString();
    }

    private float textSimilarity(
            String a,
            String b) {
        if (a == null
                || b == null
                || a.isEmpty()
                || b.isEmpty()) {
            return 0.0f;
        }

        if (a.equals(b)) {
            return 1.0f;
        }

        if (a.contains(b)
                || b.contains(a)) {
            int shorter =
                    Math.min(
                            a.length(),
                            b.length());

            int longer =
                    Math.max(
                            a.length(),
                            b.length());

            return 0.84f
                    + 0.14f
                    * shorter
                    / Math.max(
                            1.0f,
                            longer);
        }

        int maxLen =
                Math.max(
                        a.length(),
                        b.length());

        if (maxLen > 180) {
            a = a.substring(
                    0,
                    Math.min(
                            180,
                            a.length()));

            b = b.substring(
                    0,
                    Math.min(
                            180,
                            b.length()));

            maxLen =
                    Math.max(
                            a.length(),
                            b.length());
        }

        int distance =
                levenshteinDistance(
                        a,
                        b);

        return Math.max(
                0.0f,
                1.0f
                        - distance
                        / Math.max(
                                1.0f,
                                maxLen));
    }

    private int levenshteinDistance(
            String a,
            String b) {
        if (a.equals(b)) {
            return 0;
        }

        if (a.length() < b.length()) {
            String temp = a;
            a = b;
            b = temp;
        }

        int[] previous =
                new int[
                        b.length() + 1];

        int[] current =
                new int[
                        b.length() + 1];

        for (int j = 0;
             j <= b.length();
             j++) {
            previous[j] = j;
        }

        for (int i = 1;
             i <= a.length();
             i++) {
            current[0] = i;

            char ca =
                    a.charAt(
                            i - 1);

            for (int j = 1;
                 j <= b.length();
                 j++) {
                int cost =
                        ca
                                        == b.charAt(
                                                j - 1)
                                ? 0
                                : 1;

                current[j] =
                        Math.min(
                                Math.min(
                                        current[j - 1]
                                                + 1,
                                        previous[j]
                                                + 1),
                                previous[j - 1]
                                        + cost);
            }

            int[] temp =
                    previous;

            previous =
                    current;

            current =
                    temp;
        }

        return previous[
                b.length()];
    }

    private String cacheKey(
            String source) {
        return currentPairPrefix()
                + source;
    }

    private boolean isTranslationCompatible(
            String source,
            String trackKey) {
        if (source == null
                || trackKey == null) {
            return false;
        }

        return !trackKey.isEmpty();
    }

    private void translateText(
            String source) {
        final String requestPair =
                currentPairPrefix();

        final String key =
                requestPair
                        + source;

        if (translationCache.containsKey(
                key)
                || !translationPending.add(
                        key)) {
            return;
        }

        Translator localTranslator =
                translator;

        if (localTranslator == null) {
            translationPending.remove(
                    key);
            return;
        }

        localTranslator
                .translate(source)
                .addOnSuccessListener(
                        translated -> {
                            translationPending.remove(
                                    key);

                            // Ignore a callback belonging to an old language pair.
                            if (!requestPair.equals(
                                    currentPairPrefix())) {
                                return;
                            }

                            if (translated != null
                                    && !translated
                                    .trim()
                                    .isEmpty()) {
                                String clean =
                                        translated.trim();

                                translationCache.put(
                                        key,
                                        clean);

                                String requestTrackingKey =
                                        trackingKey(source);

                                synchronized (trackLock) {
                                    for (TextTrack track :
                                            tracks) {
                                        float similarity =
                                                textSimilarity(
                                                        requestTrackingKey,
                                                        track.key);

                                        if (similarity
                                                >= 0.82f) {
                                            track.translated =
                                                    clean;
                                        }
                                    }
                                }
                            }

                            refreshOverlay();

                            setStatus(
                                    "OCR + çeviri aktif • stabil takip • "
                                            + selectedSourceName()
                                            + " → "
                                            + selectedTargetName());
                        })
                .addOnFailureListener(
                        e -> {
                            translationPending.remove(
                                    key);

                            if (requestPair.equals(
                                    currentPairPrefix())) {
                                setStatus(
                                        "Çeviri bekliyor/model eksik: "
                                                + e.getMessage());
                            }
                        });
    }

    private void keepTracksAliveOnMiss() {
        long now =
                SystemClock.elapsedRealtime();

        synchronized (trackLock) {
            Iterator<TextTrack> iterator =
                    tracks.iterator();

            while (iterator.hasNext()) {
                TextTrack track =
                        iterator.next();

                track.matchedThisFrame =
                        false;

                track.missedFrames++;

                if (now - track.lastSeenAt
                        > TRACK_HOLD_MS) {
                    iterator.remove();
                }
            }
        }

        refreshOverlay();
    }

    private int activeTranslatedTrackCount(
            long now) {
        int count = 0;

        synchronized (trackLock) {
            for (TextTrack track :
                    tracks) {
                if (track.translated != null
                        && !track.translated.isEmpty()
                        && now - track.lastSeenAt
                        <= TRACK_HOLD_MS) {
                    count++;
                }
            }
        }

        return count;
    }

    private Rect displayRectForTrack(
            TextTrack track,
            long now) {
        RectF predicted =
                new RectF(
                        track.box);

        long age =
                now - track.lastSeenAt;

        if (!track.matchedThisFrame
                && age > 0
                && age <= TRACK_SOFT_PREDICT_MS
                && track.missedFrames > 0) {
            float intervalFactor =
                    Math.min(
                            2.2f,
                            age
                                    / (float)
                                    OCR_INTERVAL_MS);

            // Conservative extrapolation prevents labels from freezing for
            // one missed OCR frame while avoiding long-term drift.
            float predictStrength =
                    0.58f;

            predicted.offset(
                    track.velocityX
                            * intervalFactor
                            * predictStrength,
                    track.velocityY
                            * intervalFactor
                            * predictStrength);
        }

        return new Rect(
                Math.round(predicted.left),
                Math.round(predicted.top),
                Math.round(predicted.right),
                Math.round(predicted.bottom));
    }

    private void refreshOverlay() {
        final List<TranslationOverlayView.OverlayItem> items =
                new ArrayList<>();

        long now =
                SystemClock.elapsedRealtime();

        synchronized (trackLock) {
            for (TextTrack track :
                    tracks) {
                if (track.translated == null
                        || track.translated.isEmpty()) {
                    continue;
                }

                if (now - track.lastSeenAt
                        > TRACK_HOLD_MS) {
                    continue;
                }

                items.add(
                        new TranslationOverlayView.OverlayItem(
                                displayRectForTrack(
                                        track,
                                        now),
                                track.source,
                                track.translated));
            }
        }

        final int width =
                currentFrameWidth;

        final int height =
                currentFrameHeight;

        runOnUiThread(
                () -> overlayView.setItems(
                        items,
                        width,
                        height));
    }

    private void showModelDialog() {
        String targetCode =
                selectedTargetCode();

        String targetName =
                selectedTargetName();

        if (targetCode == null) {
            return;
        }

        String[] actions =
                new String[]{
                        "Seçili modeli indir/kontrol et",
                        "Seçili hedef modeli sil",
                        "Yüklü model sayısını göster"
                };

        new AlertDialog.Builder(this)
                .setTitle(
                        "Dil modeli • "
                                + targetName)
                .setItems(
                        actions,
                        (dialog, which) -> {
                            if (which == 0) {
                                rebuildTranslator(
                                        true);
                            } else if (which == 1) {
                                deleteTargetModel(
                                        targetCode,
                                        targetName);
                            } else {
                                showDownloadedModelCount();
                            }
                        })
                .setNegativeButton(
                        "Kapat",
                        null)
                .show();
    }

    private void deleteTargetModel(
            String languageCode,
            String languageName) {
        RemoteModelManager manager =
                RemoteModelManager.getInstance();

        TranslateRemoteModel model =
                new TranslateRemoteModel.Builder(
                        languageCode)
                        .build();

        setStatus(
                languageName
                        + " modeli siliniyor…");

        manager
                .deleteDownloadedModel(
                        model)
                .addOnSuccessListener(
                        unused -> {
                            translationCache.clear();
                            clearTrackingState();

                            setStatus(
                                    languageName
                                            + " modeli silindi.");

                            Toast.makeText(
                                            this,
                                            "Model silindi: "
                                                    + languageName,
                                            Toast.LENGTH_SHORT)
                                    .show();
                        })
                .addOnFailureListener(
                        e -> setStatus(
                                "Model silinemedi: "
                                        + e.getMessage()));
    }

    private void showDownloadedModelCount() {
        RemoteModelManager
                .getInstance()
                .getDownloadedModels(
                        TranslateRemoteModel.class)
                .addOnSuccessListener(
                        models ->
                                new AlertDialog.Builder(
                                                this)
                                        .setTitle(
                                                "Yüklü çeviri modelleri")
                                        .setMessage(
                                                "Cihazdaki ML Kit çeviri modeli sayısı: "
                                                        + models.size())
                                        .setPositiveButton(
                                                "Tamam",
                                                null)
                                        .show())
                .addOnFailureListener(
                        e -> setStatus(
                                "Model listesi alınamadı: "
                                        + e.getMessage()));
    }

    private void setStatus(
            String text) {
        runOnUiThread(
                () -> statusView.setText(
                        text));
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(
                requestCode,
                permissions,
                grantResults);

        if (requestCode
                == CAMERA_PERMISSION_REQUEST
                && grantResults.length > 0
                && grantResults[0]
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            setStatus(
                    "Kamera izni gerekli.");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }

        if (recognizer != null) {
            recognizer.close();
        }

        if (translator != null) {
            translator.close();
        }
    }

    private static class SimpleItemSelectedListener
            implements android.widget.AdapterView.OnItemSelectedListener {
        private final Runnable callback;
        private boolean first = true;

        SimpleItemSelectedListener(
                Runnable callback) {
            this.callback = callback;
        }

        @Override
        public void onItemSelected(
                android.widget.AdapterView<?> parent,
                View view,
                int position,
                long id) {
            if (first) {
                first = false;
                return;
            }

            callback.run();
        }

        @Override
        public void onNothingSelected(
                android.widget.AdapterView<?> parent) {
        }
    }
}
