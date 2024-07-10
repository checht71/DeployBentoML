# mldeployment
This is a fork of the "Deploying ML Into Production" video tutorial on The Sound of AI channel which is altered for use with PyTorch models.

[Here's the link to the video](https://www.youtube.com/watch?v=HHkmfI_yncc)
The process should be roughly the same for deploying the PyTorch model.
If you have issues with the model make sure you check your incoming datatypes vs the datatype used in the model. They need to be the exact same. Since this is based on a ResNet18 model, the batches have to be float32.
