#!/usr/bin/env bash
# Copy outside the checkout and set PHYSCANVAS_PATHS_FILE to your edited copy.
# Do not commit private dataset paths or credentials.
export PHYSCANVAS_PYTHON=/absolute/path/to/runtime/bin/python
export PHYSCANVAS_ASSETS=/absolute/path/to/physcanvas-assets
export PI05_CHECKPOINT=/absolute/path/to/downloads/policy/pi05_base
export PI05_NORM_STATS=/absolute/path/to/matching/norm_stats.json
export OPENPI_DATA_HOME=/absolute/path/to/openpi_tokenizer_cache
export VAE_PATH=/absolute/path/to/Wan2.2-TI2V-5B-Diffusers
export PROMPT_EMBEDDING=/absolute/path/to/default_prompt_embeds.pth
export CVPR2026_WM_PROMPT_EMBEDDING="$PROMPT_EMBEDDING"
export MOTION_SETTINGS=/absolute/path/to/renderer_settings.json
case "${1:?task}" in
  building_blocks|jimu)
    export RESET_CACHE=/absolute/path/to/private/building_blocks/cache
    export RM_CHECKPOINT=/absolute/path/to/private/building_blocks/rm.pt ;;
  test_cube|shiguan)
    export RESET_CACHE=/absolute/path/to/private/test_tube/cache
    export RM_CHECKPOINT=/absolute/path/to/private/test_tube/rm.pt ;;
esac
export RESET_SELECTION="$RESET_CACHE/selection_manifest.json"
