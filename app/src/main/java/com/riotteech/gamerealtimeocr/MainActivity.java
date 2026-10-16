package com.riotteech.gamerealtimeocr;

import android.Manifest;
import android.app.AlertDialog;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Rect;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

public class MainActivity extends AppCompatActivity {
    private static final int CAMERA_PERMISSION_REQUEST = 1001;
    private static final long OCR_INTERVAL_MS = 280L;

    private final LinkedHashMap<String, String> translateLanguages = new LinkedHashMap<>();
    private final Map<String, String> translationCache = Collections.synchronizedMap(
            new LinkedHashMap<String, String>(256, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<String, String> eldest) {
                    return size() > 500;
                }
            });
    private final Set<String> translationPending = Collections.synchronizedSet(new HashSet<>());

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
    private volatile List<BlockState> currentBlocks = new ArrayList<>();

    private static class BlockState {
        final Rect box;
        final String source;
        String translated;

        BlockState(Rect box, String source, String translated) {
            this.box = new Rect(box);
            this.source = source;
            this.translated = translated;
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

        sourceSpinner.setOnItemSelectedListener(new SimpleItemSelectedListener(() -> {
            configureTargetSpinner();
            rebuildRecognizer();
            rebuildTranslator(true);
            overlayView.clearItems();
        }));

        targetSpinner.setOnItemSelectedListener(new SimpleItemSelectedListener(() -> {
            rebuildTranslator(true);
            overlayView.clearItems();
        }));

        modelButton.setOnClickListener(v -> showModelDialog());

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
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
        root.addView(previewView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));

        overlayView = new TranslationOverlayView(this);
        overlayView.setClickable(false);
        root.addView(overlayView, new FrameLayout.LayoutParams(
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
        sourceSpinner.setAdapter(new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_dropdown_item,
                new ArrayList<>(Arrays.asList(
                        "English", "Türkçe", "Deutsch", "Français",
                        "Español", "Italiano", "Português",
                        "日本語", "한국어", "简体中文"))));
        row.addView(sourceSpinner, new LinearLayout.LayoutParams(0, dp(48), 1f));

        TextView arrow = new TextView(this);
        arrow.setText("  →  ");
        arrow.setTextColor(Color.WHITE);
        arrow.setTextSize(18);
        row.addView(arrow);

        targetSpinner = new Spinner(this);
        row.addView(targetSpinner, new LinearLayout.LayoutParams(0, dp(48), 1f));

        modelButton = new Button(this);
        modelButton.setText("Model");
        row.addView(modelButton, new LinearLayout.LayoutParams(dp(86), dp(48)));

        panel.addView(row);

        statusView = new TextView(this);
        statusView.setTextColor(Color.WHITE);
        statusView.setTextSize(13);
        statusView.setText("Hazırlanıyor…");
        statusView.setPadding(dp(4), dp(2), dp(4), dp(2));
        panel.addView(statusView);

        root.addView(panel, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP));

        setContentView(root);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private String selectedSourceName() {
        Object item = sourceSpinner.getSelectedItem();
        return item == null ? "English" : item.toString();
    }

    private String selectedTargetName() {
        Object item = targetSpinner.getSelectedItem();
        return item == null ? "Türkçe" : item.toString();
    }

    private String selectedSourceCode() {
        return translateLanguages.get(selectedSourceName());
    }

    private String selectedTargetCode() {
        return translateLanguages.get(selectedTargetName());
    }

    private void configureTargetSpinner() {
        String source = selectedSourceName();
        String previous = selectedTargetName();
        List<String> targets = new ArrayList<>();
        for (String language : translateLanguages.keySet()) {
            if (!language.equals(source)) targets.add(language);
        }
        targetSpinner.setAdapter(new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_dropdown_item,
                targets));
        int preferred = targets.indexOf(previous);
        if (preferred < 0) preferred = targets.indexOf("Türkçe");
        if (preferred < 0) preferred = 0;
        targetSpinner.setSelection(preferred, false);
    }

    private synchronized void rebuildRecognizer() {
        if (recognizer != null) recognizer.close();

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
            recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
        }

        translationCache.clear();
        translationPending.clear();
    }

    private synchronized void rebuildTranslator(boolean downloadIfNeeded) {
        if (translator != null) {
            translator.close();
            translator = null;
        }

        String sourceCode = selectedSourceCode();
        String targetCode = selectedTargetCode();

        if (sourceCode == null || targetCode == null || sourceCode.equals(targetCode)) {
            setStatus("Kaynak ve hedef dili farklı seçin.");
            return;
        }

        TranslatorOptions options = new TranslatorOptions.Builder()
                .setSourceLanguage(sourceCode)
                .setTargetLanguage(targetCode)
                .build();

        translator = Translation.getClient(options);
        translationCache.clear();
        translationPending.clear();

        if (downloadIfNeeded) {
            setStatus("Çeviri modeli kontrol ediliyor…");
            translator.downloadModelIfNeeded(new DownloadConditions.Builder().build())
                    .addOnSuccessListener(unused -> setStatus(
                            selectedSourceName() + " → " + selectedTargetName()
                                    + " modeli hazır • kamerayı yazıya tutun"))
                    .addOnFailureListener(e -> setStatus(
                            "Model indirilemedi: " + e.getMessage()));
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                cameraProvider = future.get();
                bindCameraUseCases();
            } catch (Exception e) {
                setStatus("Kamera açılamadı: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        if (cameraProvider == null) return;

        cameraProvider.unbindAll();

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        analysis.setAnalyzer(cameraExecutor, this::analyzeFrame);

        try {
            cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis);
            setStatus("Kamera hazır • çeviri modeli kontrol ediliyor…");
        } catch (Exception e) {
            setStatus("Kamera bağlanamadı: " + e.getMessage());
        }
    }

    @android.annotation.SuppressLint("UnsafeOptInUsageError")
    private void analyzeFrame(@NonNull ImageProxy imageProxy) {
        long now = SystemClock.elapsedRealtime();

        if (now - lastOcrAt < OCR_INTERVAL_MS || !ocrBusy.compareAndSet(false, true)) {
            imageProxy.close();
            return;
        }

        lastOcrAt = now;

        Image mediaImage = imageProxy.getImage();
        TextRecognizer localRecognizer = recognizer;

        if (mediaImage == null || localRecognizer == null) {
            ocrBusy.set(false);
            imageProxy.close();
            return;
        }

        int rotation = imageProxy.getImageInfo().getRotationDegrees();
        final int uprightWidth = (rotation == 90 || rotation == 270)
                ? imageProxy.getHeight() : imageProxy.getWidth();
        final int uprightHeight = (rotation == 90 || rotation == 270)
                ? imageProxy.getWidth() : imageProxy.getHeight();

        InputImage image = InputImage.fromMediaImage(mediaImage, rotation);
        long generation = frameGeneration.incrementAndGet();

        localRecognizer.process(image)
                .addOnSuccessListener(result -> handleRecognizedText(
                        generation,
                        result,
                        uprightWidth,
                        uprightHeight))
                .addOnFailureListener(e -> setStatus("OCR hatası: " + e.getMessage()))
                .addOnCompleteListener(task -> {
                    ocrBusy.set(false);
                    imageProxy.close();
                });
    }

    private void handleRecognizedText(
            long generation,
            Text text,
            int frameWidth,
            int frameHeight) {
        if (generation != frameGeneration.get()) return;

        List<BlockState> blocks = new ArrayList<>();

        for (Text.TextBlock block : text.getTextBlocks()) {
            Rect box = block.getBoundingBox();
            String source = normalizeText(block.getText());

            if (box == null || source.length() < 2 || box.width() < 12 || box.height() < 10) {
                continue;
            }

            String translated = translationCache.get(cacheKey(source));
            blocks.add(new BlockState(box, source, translated));
        }

        currentFrameWidth = frameWidth;
        currentFrameHeight = frameHeight;
        currentBlocks = blocks;
        refreshOverlay();

        if (blocks.isEmpty()) {
            setStatus("Yazı aranıyor…");
            return;
        }

        for (BlockState block : blocks) {
            if (block.translated == null || block.translated.isEmpty()) {
                translateBlock(generation, block.source);
            }
        }
    }

    private String normalizeText(String value) {
        if (value == null) return "";
        return value.replaceAll("\\s+", " ").trim();
    }

    private String cacheKey(String source) {
        return selectedSourceCode() + ">" + selectedTargetCode() + ">" + source;
    }

    private void translateBlock(long generation, String source) {
        final String key = cacheKey(source);

        if (translationCache.containsKey(key) || !translationPending.add(key)) return;

        Translator localTranslator = translator;
        if (localTranslator == null) {
            translationPending.remove(key);
            return;
        }

        localTranslator.translate(source)
                .addOnSuccessListener(translated -> {
                    translationPending.remove(key);

                    if (translated != null && !translated.trim().isEmpty()) {
                        translationCache.put(key, translated.trim());
                    }

                    if (generation == frameGeneration.get()) {
                        for (BlockState block : currentBlocks) {
                            String cached = translationCache.get(cacheKey(block.source));
                            if (cached != null) block.translated = cached;
                        }
                        refreshOverlay();
                        setStatus("OCR + çeviri aktif • "
                                + selectedSourceName() + " → " + selectedTargetName());
                    }
                })
                .addOnFailureListener(e -> {
                    translationPending.remove(key);
                    setStatus("Çeviri bekliyor/model eksik: " + e.getMessage());
                });
    }

    private void refreshOverlay() {
        List<TranslationOverlayView.OverlayItem> items = new ArrayList<>();

        for (BlockState block : currentBlocks) {
            if (block.translated != null && !block.translated.isEmpty()) {
                items.add(new TranslationOverlayView.OverlayItem(
                        block.box,
                        block.source,
                        block.translated));
            }
        }

        runOnUiThread(() -> overlayView.setItems(
                items,
                currentFrameWidth,
                currentFrameHeight));
    }

    private void showModelDialog() {
        String targetCode = selectedTargetCode();
        String targetName = selectedTargetName();

        if (targetCode == null) return;

        String[] actions = new String[]{
                "Seçili modeli indir/kontrol et",
                "Seçili hedef modeli sil",
                "Yüklü model sayısını göster"
        };

        new AlertDialog.Builder(this)
                .setTitle("Dil modeli • " + targetName)
                .setItems(actions, (dialog, which) -> {
                    if (which == 0) {
                        rebuildTranslator(true);
                    } else if (which == 1) {
                        deleteTargetModel(targetCode, targetName);
                    } else {
                        showDownloadedModelCount();
                    }
                })
                .setNegativeButton("Kapat", null)
                .show();
    }

    private void deleteTargetModel(String languageCode, String languageName) {
        RemoteModelManager manager = RemoteModelManager.getInstance();
        TranslateRemoteModel model = new TranslateRemoteModel.Builder(languageCode).build();

        setStatus(languageName + " modeli siliniyor…");

        manager.deleteDownloadedModel(model)
                .addOnSuccessListener(unused -> {
                    translationCache.clear();
                    setStatus(languageName + " modeli silindi.");
                    Toast.makeText(
                            this,
                            "Model silindi: " + languageName,
                            Toast.LENGTH_SHORT).show();
                })
                .addOnFailureListener(e -> setStatus(
                        "Model silinemedi: " + e.getMessage()));
    }

    private void showDownloadedModelCount() {
        RemoteModelManager.getInstance()
                .getDownloadedModels(TranslateRemoteModel.class)
                .addOnSuccessListener(models -> new AlertDialog.Builder(this)
                        .setTitle("Yüklü çeviri modelleri")
                        .setMessage("Cihazdaki ML Kit çeviri modeli sayısı: " + models.size())
                        .setPositiveButton("Tamam", null)
                        .show())
                .addOnFailureListener(e -> setStatus(
                        "Model listesi alınamadı: " + e.getMessage()));
    }

    private void setStatus(String text) {
        runOnUiThread(() -> statusView.setText(text));
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_PERMISSION_REQUEST
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            setStatus("Kamera izni gerekli.");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) cameraExecutor.shutdown();
        if (recognizer != null) recognizer.close();
        if (translator != null) translator.close();
    }

    private static class SimpleItemSelectedListener
            implements android.widget.AdapterView.OnItemSelectedListener {
        private final Runnable callback;
        private boolean first = true;

        SimpleItemSelectedListener(Runnable callback) {
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
        public void onNothingSelected(android.widget.AdapterView<?> parent) {
        }
    }
}
