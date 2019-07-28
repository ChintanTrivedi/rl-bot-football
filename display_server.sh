Xvfb :1 -screen 0 800x600x24 &
x11vnc -display :1 -localhost &

env DISPLAY=:1.0 MESA_GL_VERSION_OVERRIDE=3.2 MESA_GLSL_VERSION_OVERRIDE=150 python3 train.py

