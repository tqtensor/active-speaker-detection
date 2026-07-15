<template>
  <div class="pixelml-slide-footer" :class="{ 'footer-dark': isDarkSlide }">
    <!-- Slide number — bottom-left -->
    <span class="slide-number">{{ currentSlideNo }}</span>
    <!-- pixelml logo — bottom-right with brand clear-space compliance
         Logo: 72px wide × ~31px tall (329×143 natural, scale 0.219)
         "E" unit: rendered cap-height ≈ 19px tall, ≈ 13px wide
         Clear space rules:
           Right  ≥ 2E wide   → ≥ 26px  → padding-right: 28px
           Bottom ≥ 1.5E tall → ≥ 29px  → padding-bottom: 29px keeps logo bottom 29px from slide edge
           Top    ≥ 1.5E tall → ≥ 29px  → content padding-bottom in pixelml-brand.css reserves this
    -->
    <img
      src="/images/pixelml-short-logo.png"
      alt="pixelml"
      class="footer-logo"
      :class="{ 'logo-invert': isDarkSlide }"
    />
  </div>
</template>

<script setup>
import { useNav } from "@slidev/client";
import { computed } from "vue";

const { currentSlideNo, currentSlideRoute } = useNav();

const isDarkSlide = computed(() => {
  const cls = currentSlideRoute.value?.meta?.slide?.frontmatter?.class ?? "";
  return cls.includes("layout-section-pixelml");
});
</script>

<style scoped>
.pixelml-slide-footer {
  /* position: absolute — renders inside the slide page coordinate system
     (960×540 canvas). Using 'fixed' would escape the CSS transform/scale
     context and position relative to the browser viewport instead. */
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 4rem;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  /* padding-bottom: 29px satisfies bottom clear-space ≥ 1.5E (~29px) */
  padding: 0 1.75rem 29px;
  z-index: 100;
  pointer-events: none;
}

/* Slide number — bottom-left */
.slide-number {
  font-family: "Verdana", "Geneva", sans-serif;
  font-size: 0.65rem;
  font-weight: 600;
  color: #888;
  letter-spacing: 0.02em;
}
.footer-dark .slide-number {
  color: rgba(255, 255, 255, 0.5);
}

/* pixelml logo — bottom-right, 72px wide */
.footer-logo {
  width: 72px;
  height: auto;
  opacity: 0.85;
  filter: none;
  flex-shrink: 0;
}
.footer-logo.logo-invert {
  filter: brightness(0) invert(1);
  opacity: 0.7;
}
</style>
