package com.riotteech.gamerealtimeocr;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.text.Layout;
import android.text.StaticLayout;
import android.text.TextPaint;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.View;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TranslationOverlayView extends View {
    public static class OverlayItem {
        public final Rect sourceBox;
        public final String sourceText;
        public final String translatedText;

        public OverlayItem(Rect sourceBox, String sourceText, String translatedText) {
            this.sourceBox = new Rect(sourceBox);
            this.sourceText = sourceText == null ? "" : sourceText;
            this.translatedText = translatedText == null ? "" : translatedText;
        }
    }

    private final Paint coverPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final TextPaint textPaint = new TextPaint(Paint.ANTI_ALIAS_FLAG | Paint.SUBPIXEL_TEXT_FLAG);
    private final List<OverlayItem> items = new ArrayList<>();
    private int imageWidth = 1;
    private int imageHeight = 1;

    public TranslationOverlayView(Context context) {
        super(context);
        init();
    }

    public TranslationOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        setWillNotDraw(false);
        setLayerType(View.LAYER_TYPE_HARDWARE, null);
        coverPaint.setColor(Color.argb(220, 0, 0, 0));
        textPaint.setColor(Color.WHITE);
        textPaint.setTypeface(android.graphics.Typeface.create("sans", android.graphics.Typeface.BOLD));
    }

    public synchronized void setItems(List<OverlayItem> newItems, int frameWidth, int frameHeight) {
        items.clear();
        if (newItems != null) {
            items.addAll(newItems);
        }
        imageWidth = Math.max(1, frameWidth);
        imageHeight = Math.max(1, frameHeight);
        postInvalidateOnAnimation();
    }

    public synchronized void clearItems() {
        items.clear();
        postInvalidateOnAnimation();
    }

    public synchronized List<OverlayItem> snapshotItems() {
        return Collections.unmodifiableList(new ArrayList<>(items));
    }

    private float dp(float value) {
        return TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP,
                value,
                getResources().getDisplayMetrics());
    }

    @Override
    protected synchronized void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (items.isEmpty() || getWidth() <= 0 || getHeight() <= 0) {
            return;
        }

        float scale = Math.max(
                getWidth() / (float) imageWidth,
                getHeight() / (float) imageHeight);
        float drawnWidth = imageWidth * scale;
        float drawnHeight = imageHeight * scale;
        float offsetX = (getWidth() - drawnWidth) * 0.5f;
        float offsetY = (getHeight() - drawnHeight) * 0.5f;
        float padding = dp(5);

        for (OverlayItem item : items) {
            if (item.translatedText == null || item.translatedText.trim().isEmpty()) {
                continue;
            }

            Rect b = item.sourceBox;
            RectF mapped = new RectF(
                    offsetX + b.left * scale,
                    offsetY + b.top * scale,
                    offsetX + b.right * scale,
                    offsetY + b.bottom * scale);

            mapped.inset(-padding, -padding);
            mapped.left = Math.max(0, mapped.left);
            mapped.top = Math.max(0, mapped.top);
            mapped.right = Math.min(getWidth(), mapped.right);
            mapped.bottom = Math.min(getHeight(), mapped.bottom);

            if (mapped.width() < dp(28) || mapped.height() < dp(18)) {
                continue;
            }

            canvas.drawRoundRect(mapped, dp(5), dp(5), coverPaint);
            drawFittedText(canvas, item.translatedText.trim(), mapped);
        }
    }

    private void drawFittedText(Canvas canvas, String text, RectF box) {
        int availableWidth = Math.max(1, (int) (box.width() - dp(10)));
        int availableHeight = Math.max(1, (int) (box.height() - dp(6)));

        float low = dp(8);
        float high = Math.min(dp(38), Math.max(dp(12), box.height() * 0.62f));
        StaticLayout best = null;

        for (int i = 0; i < 7; i++) {
            float mid = (low + high) * 0.5f;
            textPaint.setTextSize(mid);
            StaticLayout layout = makeLayout(text, availableWidth);
            if (layout.getHeight() <= availableHeight && layout.getLineCount() <= 4) {
                best = layout;
                low = mid;
            } else {
                high = mid;
            }
        }

        if (best == null) {
            textPaint.setTextSize(dp(8));
            best = makeLayout(text, availableWidth);
        }

        float x = box.left + dp(5);
        float y = box.top + Math.max(0, (box.height() - best.getHeight()) * 0.5f);
        canvas.save();
        canvas.translate(x, y);
        best.draw(canvas);
        canvas.restore();
    }

    private StaticLayout makeLayout(String text, int width) {
        return StaticLayout.Builder.obtain(text, 0, text.length(), textPaint, width)
                .setAlignment(Layout.Alignment.ALIGN_CENTER)
                .setIncludePad(false)
                .setLineSpacing(0f, 1.0f)
                .build();
    }
}
