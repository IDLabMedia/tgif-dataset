# TGIF: Text-Guided Inpainting Forgery Dataset

This dataset contains approximately 75k fake images, manipulated by text-guided inpainting methods (SD2, SDXL, and Adobe Firefly).
The authentic images originate from [MS-COCO](https://cocodataset.org/), with a [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/), and have resolutions up to 1024x1024 px.
We provide both the manipulated image where the inpainted area is spliced in the original image (SD2-sp, PS-sp), as well as the fully-regenerated image (SD2-fr, SDXL-fr), when possible.

The dataset corresponds to the paper "TGIF: Text-Guided Inpainting Forgery Dataset", which is currently submitted and under review.

We distribute this dataset under the [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).

![TGIF Creation](./readme-images/TGIF_diagram.png)

## TODOs
- [ ] Add benchmark results (per image)
- [ ] Add code for image compression
- [ ] Add paper pdf and BibTex code

## Dataset specifications
| **Manipulation types**                             |                                    |
|----------------------------------------------------|------------------------------------|
| **# masks**                                        | 2 (segmentation & bounding box)    |
| **# variations** (num_images_per_prompt)           | 3 per generation                   |
| **# sub-datasets**                                 | 4 (SD2-sp, PS-sp, SD2-fr, SDXL-fr) |
| **Total # manipulated images per authentic image** | 2 * 3 * 4 = 24                     |

| **Dataset size**         | **Training** | **Validation** | **Testing** | **Total** |
|--------------------------|--------------|----------------|-------------|-----------|
| **# authentic images**   | 2 440        | 341            | 343         | 3 124     |
| **# manipulated images** | 58 560       | 8 184          | 8 232       | 74 976    |

## Downloadlinks
[Download all images](https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o)

The downloads are organized in masks, original, SD2-sp, PS-sp, SD2-fr, SDXL-fr. And each of those are separated in training, validation, and testing, respectively.

