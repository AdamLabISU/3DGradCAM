# 3DGradCAM
This is a repository for 3D Gradient weighted Class Activation Maps for Manufacturability Analysis. For more details, please see the paper: 
**[Learning Localized Geometric Features Using 3D-CNN: An Application to Manufacturability Analysis of Drilled Holes][1]**  
Aditya Balu, Sambit Ghadai, Kin Gwn Lore, Gavin Young, Adarsh Krishnamurthy, Soumik Sarkar  

Demo: [Adam Lab][2]
<video controls="controls">
  <source type="video/mp4" src="demo.mp4"></source>
</video>

## Packages needed
  Keras 2.0.5, Tensorflow 1.2

## Usage
You can either import the GradCAM function:

    from GradCAM import prepareGradCAM, GradCAM
    activationFunction=prepareGradCAM(cnnModel, layerIdx, nbClasses)

activationFunction is a Keras Function for computing the class activation map

## License
BSD

### Future work
The following codes shall be made available soon:
1. Data Generation using ACIS solid modeling kernel
2. GPU Accelerated Voxelization: Available now @ https://github.com/idealab-isu/GPView
3. Training
4. GPU accelerated Volume Rendering of GradCAM output

[1]: https://arxiv.org/abs/1612.02141
[2]: http://web.me.iastate.edu/adamlab/r-manufacturability.html
