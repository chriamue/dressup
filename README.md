# dressup
Shows clothes given as images at persons on webcam image.

Example for openpose and test for usage of openpose and cmake.

## scarf
Icon made by [Madebyoliver](https://www.flaticon.com/authors/madebyoliver) from www.flaticon.com

## skirt
Icon made by [Freepik](https://www.flaticon.com/authors/freepik) from www.flaticon.com

## tshirt
Icon made by [Freepik](https://www.flaticon.com/authors/freepik) from www.flaticon.com

## run as docker
'''bash
xhost +
nvidia-docker run --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v '/tmp/.X11-unix:/tmp/.X11-unix' chriamue/dressup
'''