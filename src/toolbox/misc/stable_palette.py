def stable_palette(labels):
    # Predefined palette.
    # colors = ['#4974a5', '#ffa500', '#5d782e', '#545454', '#d7b4ae', '#00ff00', '#00ffff', '#6fa287', \
    #           '#ee30a7', '#9e482f', '#8b02e7', '#141387', '#eccb00', '#ff0000', '#ff9664', '#b73e64', \
    #           '#0000ff', '#969696', '#969600', '#ffe600', '#ff6400', '#1f1e33']
    # A combination of Dutch Field, River Nights, and Spring Pastels from https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data
    colors = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0",
              "#b30000", "#7c1158", "#4421af", "#1a53ff", "#0d88e6", "#00b7c7", "#5ad45a", "#8be04e", "#ebdc78",
              "#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]

    if len(labels) >= len(colors):
        raise ValueError('Too many labels.')

    return dict(zip(labels, colors))