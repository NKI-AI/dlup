Transforms
==========

The transforms can be used for several histopathology relevant augmentations.
The dataset classes output data in the form

.. code-block:: javascript

    {
       "image": either string or some image array type
       "mask": image label, of the same shape as image
       "bbox": bounding box
       "points": points
       "contour": polygon
       "class_label": whole slide image label
    }

We wrap several existing libraries into the format dlup requires.


Torchvision transforms
----------------------
:code:`torchvision` transforms, when they would only influence the image content and not the mask, are
wrapped and prefixed by :code:`Tv`. For instance :code:`dlup.transforms.TvGrayscale`.

These torchvision transforms are wrapped:
    :code:`LinearTransformation`,
    :code:`ColorJitter`,
    :code:`Grayscale`,
    :code:`Normalize`,
    :code:`CenterCrop`,
    :code:`Pad`,
    :code:`FiveCrop`,
    :code:`TenCrop`,
    :code:`RandomGrayScale`,
    :code:`RandomErasing`,
    :code:`RandomInvert`,
    :code:`RandomPosterize`,
    :code:`RandomSolarize`,
    :code:`RandomAdjustSharpness`,
    :code:`RandomAutocontrast`, and
    :code:`RandomEqualize`.

These transforms are also wrapped, but they do not support masks, contours, points or bounding boxes as they have
a random component that changes image geometry:

    :code:`TvRandomCrop`,
    :code:`TvRandomHorizontalFlip`,
    :code:`TvRandomVerticalFlip`,
    :code:`TvRandomPerspective`,
    :code:`TvRandomRotation`, and
    :code:`TvRandomAffine`.


HistomicsTK transforms
----------------------
:code:`histomicstk` transforms, when they would only influence the image content and not the mask, are
wrapped in a class (name converted to CamelCase) and prefixed by :code:`Htk`.
For instance :code:`dlup.transforms.HtkRgbPerturbStainConcentration`.
