# TGIF: Text-Guided Inpainting Forgery Dataset

This dataset contains approximately 75k fake images, manipulated by text-guided inpainting methods (SD2, SDXL, and Adobe Firefly).
The authentic images originate from [MS-COCO](https://cocodataset.org/), with a [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/), and have resolutions up to 1024x1024 px.
We provide both the manipulated image where the inpainted area is spliced in the original image (SD2-sp, PS-sp), as well as the fully-regenerated image (SD2-fr, SDXL-fr), when possible.

The dataset corresponds to the paper "TGIF: Text-Guided Inpainting Forgery Dataset", which was accepted at the [IEEE International Workshop on Information Forensics & Security 2024](https://wifs2024.uniroma3.it/).

An extended version (TGIF2) is currently under review.

We distribute this dataset under the [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).


## Visual explanation of our insights
![Authors on skis in Greece](./readme-images/fake-skis.png)<br>
*Did the authors really go skiing on Greece's iconic Mt. Athos?*

The image above is fake - the skis were added using text-guided inpainting. Can current forensic methods detect this manipulation?

Find out in our [**blog post**](https://media.idlab.ugent.be/tgif-dataset), where we explain our insights in a simple and visual way.


## Dataset specifications

![TGIF Creation](./readme-images/TGIF_diagram.png)<br>
*In TGIF, we created 75k fake images using SD2, SDXL, and Adobe Photoshop/Firefly. We used 2 types of masks, and differentiate between spliced and fully regenerated inpainted images. Not seen in the diagram: each inpainting operation creates 3 variations in batch.*

*In TGIF2, we created an additional 140k fake images using FLUX.1 schnell, FLUX.1 dev, FLUX.1 filldev with the same 2 types of masks, as well as SD2 and SDXl with a random 64x64 px patch as mask.*

| **Manipulation types**                             |                                    |
|----------------------------------------------------|------------------------------------|
| **# masks**                                        | 2 (segmentation & bounding box) or 1 (random 64x64 px patch)   |
| **# sub-datasets**                                 | 4 (SD2-sp, PS-sp, SD2-fr, SDXL-fr) |
| **# sub-datasets (+TGIF2)**                                 | 6+3 (flux1schnell-sp, flux1schnell-fr, flux1dev-sp, flux1dev-fr, flux1filldev-sp, flux1filldev-fr) + (sd2-random-sp, sd2-random-fr, sdxl-random-fr) |
| **# variations** (num_images_per_prompt)           | 3 per generation (in batch)        |
| **Total # manipulated images per authentic image (TGIF)** | 2 * 4 * 3 = 24                     |
| **Total # manipulated images per authentic image (+TGIF2)** | (2 * 6 * 3) + (1 * 3 * 3) = 45                     |

| **Dataset size**         | **Training** | **Validation** | **Testing** | **Total** |
|--------------------------|--------------|----------------|-------------|-----------|
| **# authentic images**   | 2 440        | 341            | 343         | 3 124     |
| **# manipulated images (TGIF)** | 58 560       | 8 184          | 8 232       | 74 976    |
| **# manipulated images (+TGIF2)** | 109 800       | 15 345          | 15 435       | 140 580    |
| **# manipulated images (TGIF+TGIF2)** | 168 360       | 23 529          | 23 667       | 215 556    |


## Download
* [Download all TGIF images](https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o)
* [Download all TGIF2 FLUX images](https://cloud.ilabt.imec.be/index.php/s/KG48tLZZzifC5WE)
* [Download all TGIF2 random images](https://cloud.ilabt.imec.be/index.php/s/CyPBBQPiFM55JRF)

For TGIF, the downloads are organized in masks, original, SD2-sp, PS-sp, SD2-fr, SDXL-fr. 
For TGIF2 FLUX, the downloads are organized in flux1schnell-sp, flux1schnell-fr, flux1dev-sp, flux1dev-fr, flux1filldev-sp, flux1filldev-fr. They additionally *but redundantly* contain masks-flux, original-flux. You can also use the masks and original from TGIF instead.
For TGIF2 random, the downloads are organized in masks, masks-fr, sd2-sp, sd2-fr, sdxl-fr.
Each of the directories mentioned above are separated in training, validation, and testing, respectively.

Metadata (incl. NIMA, GIQA & ITM scores) is available in this repository (_metadata_, _metadata\_flux_, and _metadata\_random_). Benchmark results of TGIF (_benchmark-results_).

Code to perform text-guided inpainting with SD2, SDXL & Adobe Photoshop/Firefly is added in the _code_ folder of this repository, as well as code to calculate NIMA, GIQA, and ITM scores, and to compress images using JPEG and WEBP.

The [NIMA](https://github.com/yunxiaoshi/Neural-IMage-Assessment) and [GIQA](https://github.com/cientgu/GIQA) checkpoints are archived [here](https://cloud.ilabt.imec.be/index.php/s/DQwTjE6A3ydgxjq). The ITM, SD2 and SDXL weights are downloaded automatically.

## Filenaming
The files are named as follows:
* orig:
  * {coco_id}_orig.png
  * {coco_id}\_orig\_{crop_size}.png
* masks:
  * {coco_id}\_mask\_{crop_size}.png
  * {coco_id}\_mask\_{mask_type}.png
  * * {coco_id}\_mask\_{mask_type}\_{crop_size}.png
  * {coco_id}\_mask\_{mask_type}.png\_ps\_mask.png - Photoshop adaptation of mask (extra border)
* SD2-sp, flux1schnell-sp, flux1dev-sp, flux1filldev-sp: {coco_id}\_mask\_{mask_type}.png_ps_mask.png_{gen_model}_{var_id}.png
* PS-sp: {coco_id}\_mask\_{mask_type}.png_ps_{var_id}.png (i.e., no extra ps_mask.png in filename)
* SD2-fr: {coco_id}\_mask\_{mask_type}.png_sd2-512_{var_id}.png  (i.e., 512 instead of 1024)
* SDXL-fr, flux1schnell-fr, flux1dev-fr, flux1filldev-fr: {coco_id}\_mask\_{mask_type}.png_{gen_model}-1024_{var_id}.png (i.e., 1024 instead of 512)
* SD2-random-sp: {coco_id}\_mask\_random.png_sd2_{var_id}.png
* SD2-random-fr: {coco_id}\_mask\_random.png_sd2-512_{var_id}.png
* SDXL-random-fr: {coco_id}\_mask\_random.png_sd2-1024_{var_id}.png

With
* crop_size: 512 or 1024
* mask_type: bbox, segm, or random
* var_id: 0, 1, or 2
* gen_model: sd2, sdxl, flux1schnell, flux1dev, flux1filldev

## Reference
TGIF was presented in the [IEEE International Workshop on Information Forensics & Security 2024](https://wifs2024.uniroma3.it/). The preprint can be downloaded [on arXiv](https://arxiv.org/abs/2407.11566), and the published version on [IEEEXplore](https://ieeexplore.ieee.org/document/10810690).

```js
@InProceedings{mareen2024tgif,
  author="Mareen, Hannes and Karageorgiou, Dimitrios and Van Wallendael, Glenn and Lambert, Peter and Papadopoulos, Symeon",
  title="TGIF: Text-Guided Inpainting Forgery Dataset",
  booktitle="Proc. Int. Workshop on Information Forensics and Security (WIFS) 2024",
  year="2024"
}
```
