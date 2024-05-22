# ComfyUI MuseV Evolved

Based on the diffusion model, let us Anymate anything.

> This is the ComfyUI version of [MuseV](https://github.com/TMElyralab/MuseV), which also draws inspiration from [ComfyUI-MuseV](https://github.com/chaojie/ComfyUI-MuseV). It offers more configurable parameters, making it more flexible in implementation.

![example01](./docs/assets/example00.gif) 

|  Input   | Output  | 
|  ----  | ----  | 
| ![example01](./docs/assets/example01_input.png) | ![example01](./docs/assets/example01_output.gif) | 
| ![example02](./docs/assets/example02_input.png) | ![example02](./docs/assets/example02_output.gif) | 
| ![example03](./docs/assets/example03_input.webp) | ![example02](./docs/assets/example03_output.gif) | 

## Example Workflow

|  Input   | Output  | Workflow Preview | Workflow |
|  ----  | ----  | ---- | ---- |
| ![example01](./docs/assets/example01_input.png) | ![example01](./docs/assets/example01_output.gif) | ![example01](./docs/assets/example01_workflow.png) | [workflow](./docs/assets/example01_workflow.json) |


## Prepare Models

download https://huggingface.co/TMElyralab/MuseV to ComfyUI/models/diffusers

```
huggingface-cli download --resume-download TMElyralab/MuseV --local-dir ComfyUI/models/diffusers/TMElyralab/MuseV
```

## Contribution

Thank you for considering to help out with the source code! Welcome contributions from anyone on the internet, and are grateful for even the smallest of fixes!

If you'd like to contribute to this project, please fork, fix, commit and send a pull request for me to review and merge into the main code base.