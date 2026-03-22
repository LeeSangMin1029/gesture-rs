#!/bin/bash
# Gesture Control - run script
export ORT_DYLIB_PATH="C:\Users\skswo\Desktop\project\gesture-rs\onnxruntime.dll"
export RUST_LOG=gesture_control=debug
cargo run "$@"
