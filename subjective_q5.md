# Question - 5

# Part-1

1. **Region is a rectangular block of 30x30 is assumed missing from the image.**

    - **Using Gradient Descent:**
    
        <img src="figures\part1_30x30.png" width="900" height="300">

        **Observation:** The reconstruction quality is poor in the area where the rectangular block was removed because the algorithm lacks sufficient contextual information to accurately predict missing pixels deep within the removed patch. The quality is slightly better near the boundary pixels of the removed patch because there is some local information available from neighboring pixels. However, as we move deeper into the removed patch, the available information diminishes, leading the algorithm to stretch information from the border pixels. If the missing patch contains only a single color, the reconstruction is relatively good. But if the missing patch contains multiple colors, the reconstruction becomes poorer as the algorithm struggles to accurately fill in the missing information.
    - **Reconstruction using RFF + Linear regression:**
    
        <div style="text-align: center;">
            <img src="figures/Patch_contaning_single_color_using_RFF(300x300).png" alt="Reconstruction using RFF" width="500" height="300">
        </div>


2. **A random subset of 900 (30x30) pixels is missing from the image.**

    - **Using Gradient Descent:**
    
        <img src="figures/Randomly_removed_patchs_reconstruction.png" alt="Random Subset Reconstruction" width="900" height="300">

        **Observation:** When small portions of pixels are removed randomly and we use gradient descent to factorize matrices W and H with an appropriate rank for matrix decomposition, there's a high likelihood that the removed pixels will have at least one neighboring pixel that is not removed. This means that some information about the removed pixels is retained, resulting in better reconstruction compared to when a whole patch of pixels is removed. This is because the algorithm can use the surrounding pixels to infer missing values, leading to a smoother and more accurate reconstruction.    

    - **Reconstruction using RFF + Linear regression:**
    
        <div style="text-align: center;">
            <img src="figures/usingrandomsubset.png" alt="Reconstruction using RFF" width="500" height="300">
        </div>

# Part - 2

***Vary region size (NxN) for N = [20, 40, 60, 80] and perform Gradient Descent till convergence.***

1. **Rectangular block of size NxN is assumed to be missing from the image:**

    - **Region of 20x20:**
    
        <img src="figures/part1patch(20x20).png" alt="20x20 Missing Patch" width="900" height="300">

    - **Region of 40x40:**
    
        <img src="figures/part1patch(40x40).png" alt="40x40 Missing Patch" width="900" height="300">

    - **Region of 60x60:**
    
        <img src="figures/part1patch(60x60).png" alt="60x60 Missing Patch" width="900" height="300">

    - **Region of 80x80:**
    
        <img src="figures/part1patch(80x80).png" alt="80x80 Missing Patch" width="900" height="300">

    **Observation:** As we increase the size N of the patch to be removed while keeping the rank r constant, the reconstruction quality tends to deteriorate. This is because with a larger patch size, there's a greater proportion of missing pixels within the image. This creates a larger area of missing information, making it more difficult for the algorithm to accurately fill in the gaps. Additionally, as the patch size increases, the algorithm has to interpolate a greater number of missing pixels, which reduces the amount of contextual information available from neighboring pixels. Consequently, the reconstruction becomes poorer in the region of the removed pixels, as the algorithm struggles to capture the complex patterns and structures present in the image. However, it's important to note that the reconstruction quality remains good in the regions of the image that are not affected by the removal of pixels.

2. **Random subset of size NxN is assumed to be missing from the image:**

    - **Region of 20x20:**
    
        <img src="figures/Randomlyremoved20x20pixelsreconstruction.png" alt="Randomly Removed 20x20" width="900" height="300">

    - **Region of 40x40:**
    
        <img src="figures/Randomlyremoved40x40pixelsreconstruction.png" alt="Randomly Removed 40x40" width="900" height="300">

    - **Region of 60x60:**
    
        <img src="figures/Randomlyremoved60x60pixelsreconstruction.png" alt="Randomly Removed 60x60" width="900" height="300">

    - **Region of 80x80:**
    
        <img src="figures/Randomlyremoved80x80pixelsreconstruction.png" alt="Randomly Removed 80x80" width="900" height="300">

## PSNR (Peak Signal-to-Noise Ratio)

PSNR is a measure used to assess the quality of a reconstructed or compressed image relative to the original image. It quantifies the ratio between the maximum possible power of a signal and the power of the noise that affects the fidelity of its representation.

### Formula
The formula for PSNR is:

$$
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{{\text{MAX}^2}}{{\text{MSE}}} \right)
$$

Where:
- **MAX**: Maximum possible pixel value of the image (e.g., 255 for an 8-bit grayscale image).
- **MSE**: Mean Squared Error, calculated as the average of the squared differences between corresponding pixels in the original and reconstructed images.

### Interpretation
PSNR is typically expressed in decibels (dB). A higher PSNR value indicates higher image quality, meaning less distortion or noise in the reconstructed image compared to the original.

### PSNR Graph of Randomly Removed Pixels and Missing Patch

<div style="text-align: center;">
    <img src="figures/psnr.png" alt="Reconstruction using RFF" width="400" height="300">
</div>

- **Observation**:From the PSNR graph comparing randomly removed points and patch removed points, we observe that the PSNR value for randomly removed points is higher. As we increase the size of the removed points (NxN) from 20 to 80, the PSNR value decreases slightly. However, for patch removed points, the PSNR value for subsequent sizes is smaller compared to randomly removed points. Moreover, as we increase the size (N) of the removed patch, the PSNR value decreases significantly, indicating poorer reconstruction quality.

- **Conclusion**:

    1. Randomly removing points results in higher PSNR values, indicating better reconstruction quality compared to patch removal.

    2. Increasing the size of the removed patch leads to a significant decrease in PSNR values compare to randomly removed points, suggesting a notable decline in reconstruction quality.


## Root Mean Squared Error (RMSE)

RMSE, or Root Mean Squared Error, is a metric used to measure the average difference between corresponding pixel intensities in the original image and the reconstructed image. It provides a single numerical value that quantifies the overall discrepancy between the two images.

### Formula
The formula for RMSE is:

$$
RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (I_{\text{orig}}(i) - I_{\text{recon}}(i))^2}
$$

Where:
- **N**: Total number of pixels in the image.
- **I_orig(i)**: Intensity of the i-th pixel in the original image.
- **I_recon(i)**: Intensity of the i-th pixel in the reconstructed image.

### Interpretation
- **Lower RMSE**: A lower RMSE value indicates that the reconstructed image closely matches the original image, with minimal differences between pixel intensities. This suggests a high level of accuracy in the reconstruction process and reflects a high-quality reconstruction.
  
- **Higher RMSE**: Conversely, a higher RMSE value indicates a greater discrepancy between the original and reconstructed images. This suggests that the reconstruction process has not accurately captured the details or patterns present in the original image, resulting in a lower-quality reconstruction.

### RMSE Graph of Randomly Removed Pixels and Missing Patch

<div style="text-align: center;">
    <img src="figures/rmse.png" alt="Reconstruction using RFF" width="400" height="300">
</div>

- **Observation**: From the RMSE plots comparing patch (NxN) removed and randomly removed points (of size NxN), it's evident that the RMSE value for the patch removed is higher than for randomly removed points. As we increase the size (N) of the removed patch, the RMSE value increases significantly, indicating a substantial deterioration in reconstruction quality. On the other hand, for randomly removed points, the RMSE value remains almost constant as N increases. This suggests that the reconstruction quality for randomly removed points is relatively stable across different sizes, whereas the reconstruction quality for patch removed points deteriorates significantly as the size of the removed patch increases.

- **Conclusion**:

    1. Patch removal results in poorer reconstruction quality compared to randomly removed points.

    2. Larger removed patch sizes lead to significantly higher RMSE values, emphasizing the detrimental effect of larger missing areas on reconstruction quality.




# Part - 3
## Part -1
***Using least squre function***

1. **Region is a rectangular block of 30x30 is assumed missing from the image.**

    - **Using Gradient Descent:**
    
        <img src="figures/part1patch(30x30)ls.png" alt="30x30 Missing Patch" width="900" height="300">


2. **A random subset of 900 (30x30) pixels is missing from the image.**

    - **Using Gradient Descent:**
    
        <img src="figures/Randomly_removed_patchs_reconstruction_ls.png" alt="Random Subset Reconstruction" width="900" height="300">


## Part - 2

***Vary region size (NxN) for N = [20, 40, 60, 80] and perform Gradient Descent till convergence.***

1. **Rectangular block of size NxN is assumed to be missing from the image:**

    - **Region of 20x20:**
    
        <img src="figures/part1patch(20x20)_ls.png" alt="20x20 Missing Patch" width="900" height="300">

    - **Region of 40x40:**
    
        <img src="figures/part1patch(40x40)_ls.png" alt="40x40 Missing Patch" width="900" height="300">

    - **Region of 60x60:**
    
        <img src="figures/part1patch(60x60)_ls.png" alt="60x60 Missing Patch" width="900" height="300">

    - **Region of 80x80:**
    
        <img src="figures/part1patch(80x80)_ls.png" alt="80x80 Missing Patch" width="900" height="300">

2. **Random subset of size NxN is assumed to be missing from the image:**

    - **Region of 20x20:**
    
        <img src="figures/Randomlyremoved20x20pixelsreconstruction_ls.png" alt="Randomly Removed 20x20" width="900" height="300">

    - **Region of 40x40:**
    
        <img src="figures/Randomlyremoved40x40pixelsreconstruction_ls.png" alt="Randomly Removed 40x40" width="900" height="300">

    - **Region of 60x60:**
    
        <img src="figures/Randomlyremoved60x60pixelsreconstruction_ls.png" alt="Randomly Removed 60x60" width="900" height="300">

    - **Region of 80x80:**
    
        <img src="figures/Randomlyremoved80x80pixelsreconstruction_ls.png" alt="Randomly Removed 80x80" width="900" height="300">
### Alternating Least Square Method
We use least square funtion using `torch.linalg.lstsqr`.
### Observation 
Here, we can see from the image that the reconstructed least square method is not so good and it is not giving exact image. This is happening because of the fact that the alternating least square method just tries to minimize the error between the original image with patch removed in it and the reconstructed image. But, it does not take into account the missing patch and tries to fit the rest of the image without it. This is the reason why the reconstructed image is not so good. So from the image you can see that the prediction made for missing part of the image is not that much good and it is not able to predict the exact image.

Due to overfitting we see that this method does not predict the trend properly because of overfitting and as you can notice the part which was not cut was predicted very good and this is happening because only that part is going under training and we are predicting the values for the missing part but when we give the training part to the least square function with high number of features then it tries to fit the training data very well and hence it overfits the data and hence the prediction is not good. So we tried to decrease the number of features but in that case the the whole image coming out was not good so we kept the number of features to the current value where it was giving the best result.

Also, the time taken by the alternating least square method is very high as compared to the gradient descent method. This is because the alternating least square method is not so efficient and it takes a lot of time to converge to the optimal solution as we seen during the lectures that gradient descent is very fast as compared to the alternating least square method.

However the iteration needed to converge are very less because in each iteration it gives most optimal W and H that fits the image with missing patch very well and there is very negligible decrease in loss after 2-3 iterations.  
# Part - 4

## Regions Containing Mainly Single Color

### Vary rank (r) for r = [5, 10, 25, 50] and perform Gradient Descent till convergence.

#### 1. r = 5    
<img src="figures/part_4_im1.png" alt="Randomly Removed 20x20" width="1200" height="300">

#### 2. r = 10    
<img src="figures/part_4_im2.png" alt="Randomly Removed 20x20" width="1200" height="300">

#### 3. r = 25    
<img src="figures/part_4_im3.png" alt="Randomly Removed 20x20" width="1200" height="300">

#### 4. r = 50    
<img src="figures/part_4_im4.png" alt="Randomly Removed 20x20" width="1200" height="300">

- **Observation**: As we increase the rank (r) for matrix factorization, the reconstruction quality improves, as evidenced by the reduction in artifacts and the better preservation of the original image's structure and patterns. This is because a higher rank allows for a more accurate capture of the underlying structure and patterns in the image, leading to a more faithful reconstruction. Additionally, the PSNR and RMSE values improve as the rank increases, indicating better reconstruction quality and a closer match between the original and reconstructed images.

### PSNR and RMSE for Regions Containing Mainly Single Color

<div style="text-align: center;">
    <img src="figures/part_4_im1_psnr_rmse.png" alt="Reconstruction using RFF" width="700" height="300">
</div>

- **Observation**: The PSNR and RMSE plots show that as the rank (r) increases, the PSNR value improves, indicating better reconstruction quality. Similarly, the RMSE value decreases, reflecting a closer match between the original and reconstructed images. This suggests that a higher rank leads to a more accurate capture of the underlying structure and patterns in the image, resulting in a higher-quality reconstruction.

- **Conclusion**:

    1. Increasing the rank (r) for matrix factorization leads to better reconstruction quality for regions containing mainly single colors.

    2. Higher ranks result in improved PSNR and RMSE values, indicating a closer match between the original and reconstructed images and better preservation of the image's structure and patterns.

## Regions Containing 2-3 Different Colors

### Vary rank (r) for r = [5, 10, 25, 50] and perform Gradient Descent till convergence.

#### 1. r = 5    
<img src="figures/part_4_im5.png" alt="Randomly Removed 20x20" width="1200" height="300">

#### 2. r = 10    
<img src="figures/part_4_im6.png" alt="Randomly Removed 20x20" width="1200" height="300">

#### 3. r = 25    
<img src="figures/part_4_im7.png" alt="Randomly Removed 20x20" width="1200" height="300">

#### 4. r = 50    
<img src="figures/part_4_im8.png" alt="Randomly Removed 20x20" width="1200" height="300">

### PSNR and RMSE for Regions Containing 2-3 Different Colors

<div style="text-align: center;">
    <img src="figures/part_4_im2_psnr_rmse.png" alt="Reconstruction using RFF" width="700" height="300">
</div>

## Regions Containing at Least 5 Different Colors

### Vary rank (r) for r = [5, 10, 25, 50] and perform Gradient Descent till convergence.

#### 1. r = 5    
<img src="figures/part_4_im9.png" alt="Randomly Removed 20x20" width="1200" height="300">

#### 2. r = 10    
<img src="figures/part_4_im10.png" alt="Randomly Removed 20x20" width="1200" height="300">

#### 3. r = 25    
<img src="figures/part_4_im11.png" alt="Randomly Removed 20x20" width="1200" height="300">

#### 4. r = 50    
<img src="figures/part_4_im12.png" alt="Randomly Removed 20x20" width="1200" height="300">

### PSNR and RMSE for Regions Containing at Least 5 Different Colors

<div style="text-align: center;">
    <img src="figures/part_4_im3_psnr_rmse.png" alt="Reconstruction using RFF" width="700" height="300">
</div>