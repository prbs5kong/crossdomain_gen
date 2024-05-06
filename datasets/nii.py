import os
import nibabel as nib
import matplotlib.pyplot as plt

def convert_nii_to_png(input_folder, output_folder):
    # make sure output_folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # iterate through all .nii files in input_folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii"):
            # form full path for .nii file
            input_file = os.path.join(input_folder, filename)
            
            img = nib.load(input_file)
            data = img.get_fdata()
            data_max = data.max()
            
            # make a subfolder for each .nii file
            subfolder = os.path.splitext(filename)[0]  # remove .nii extension
            out_subfolder = os.path.join(output_folder, subfolder)
            os.makedirs(out_subfolder, exist_ok=True)
            
            # iterate through each slice and save as .png
            for slice_num in range(data.shape[2]):
                plt.imshow(data[:, :, slice_num] / data_max, cmap='gray')
                plt.axis('off')
                plt.savefig(os.path.join(out_subfolder, f"slice_{slice_num}.png"), 
                            bbox_inches='tight', pad_inches=0)
                plt.close()

convert_nii_to_png('./IXI', './IXI_image')