import os
if __name__ == "__main__":
    output_path = "/data2/tiancai/code_new_model/RepDETR/"
    dir = "/data2/tiancai/datasets/DOTAV1_0/val/images/"
    file = os.listdir(dir)
    with open(os.path.join(output_path, "../imgnamefilev1_0.txt"), 'a') as f:
        for imagename in file:
            imagename =imagename.split(".")[0]
            f.write(str(imagename))
            f.write("\n")