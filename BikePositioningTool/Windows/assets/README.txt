Background photos for the UI (optional)
=======================================

Drop image files here to skin the app. Both are optional - if missing, the app
uses a clean flat dark theme instead.

  gui_bg.jpg         -> header banner of the Tkinter control panel
                        (the wide bike-on-black photo works great here)

  dashboard_bg.jpg   -> full-page background of the analysis dashboard
                        (the close-up frame/headset photo works well)
                        .png also accepted (dashboard_bg.png)

Notes
- The dashboard embeds its background into analysis.html as base64, so the HTML
  stays self-contained and portable.
- The GUI banner needs Pillow (already installed). It auto-scales/crops the
  image to the banner size and darkens the left side so the title stays legible.
- Any resolution works; landscape suits the banner, the dashboard covers fully.
