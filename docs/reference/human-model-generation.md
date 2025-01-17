## human_model_generation module

The *human_model_generation* module contains the *PIFuGeneratorLearner* class, which inherits from the abstract class *Learner*.

### Class PIFuGeneratorLearner
Bases: `engine.learners.Learner`

The *PIFuGeneratorLearner* class is a wrapper of the PIFu [[1]](#pifu-paper) object detection algorithm based on the original
[PIFu implementation](https://github.com/shunsukesaito/PIFu).
It can be used to perform human model generation from a single image (inference). In addtion, the *PIFuGeneratorLearner* enables the 3D pose approximation of a generated human model as well as the generation of multi-view renderings of the human model.

The [PIFuGeneratorLearner](#src.opendr.simulation.human_model_generation.pifu_generator_learner.py ) class has the
following public methods:

#### `PIFuGeneratorLearner` constructor
```python
PIFuGeneratorLearner(self, device, checkpoint_dir)
```

Constructor parameter explanation:
- **device**: *{'cuda', 'cpu'}, default='cpu'*\
Specifies the device to be used.
- **checkpoint_dir**: *str, default='utilities/PIFu/checkpoints'*\
Specifies a path to be used for loading the checkpoints for inference. 
  
#### `PIFuGeneratorLearner.infer`
```python
PIFuGeneratorLearner.infer(self, imgs_rgb, imgs_msk, obj_path, extract_pose)
```

This method generates a 3D human model from a single image.
A future release will allow the use of multiple images as input.
The joints of the 3D model in the 3D space can be optionally approximated.
The method returns the 3D human model as an object of type *simulation.human_model_generation.utilities.model_3D.Model_3D*.
The locations of the model's joints are also returned as a List, in case were computed.

Parameters:
- **imgs_rgb**: *list, default=None*\
List of images of type *engine.data.Image*. These images will be used as input. At the current release, the list's length must be 1. 
- **imgs_msk**: *list, default=None*\
List of images of type *engine.data.Image*. These images will be used as masks, depicting the silhouette of the portrayed human. At the current release, the list's length must be 1. 
- **obj_path**: *str, default=None*\
Specifies a path for saving the generated 3D human model in OBJ format.
 - **extract_pose**: *bool, default=False*\
Specifies whether the joints of the 3D model in the 3D space will be approximated or not.

#### `PIFuGeneratorLearner.load`
```python
PIFuGeneratorLearner.load(self, path)
```  

This method loads a pretrained model.

Parameters:
- **path**: *str, default=None**\
Specifies the folder where the model will be loaded from.
  
#### `PIFuGeneratorLearner.download`
```python
PIFuGeneratorLearner.download(self, path, url)
```  

This method downloads the needed pretrained models.

Parameters:
- **path**: *str, default=None*\
Specifies the folder where data will be downloaded.
- **url**: *str, default=OpenDR FTP URL*\
URL of the FTP server.
  
#### References
<a name="pifu-paper" href="https://shunsukesaito.github.io/PIFu/">[1]</a>
PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization,
[arXiv](https://arxiv.org/abs/1905.05172).  
